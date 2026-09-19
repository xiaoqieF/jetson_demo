from collections import deque
import threading
import time
from typing import Any

import rclpy
from argus_interfaces.action import QwenInference
from argus_interfaces.msg import ArgusInferenceResult
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from .synchronizer import stamp_to_nanoseconds


SENSOR_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE)


class OperatorNode(Node):
    def __init__(self, bridge: Any, processor: Any) -> None:
        super().__init__("argus_operator_ui")
        self._bridge, self._processor = bridge, processor
        self.declare_parameter("image_topic", "/camera/image/compressed")
        self.declare_parameter("result_topic", "/camera/inference/result")
        self.declare_parameter("qwen_action_name", "/camera/inference/qwen")
        self.declare_parameter("image_timeout_sec", 2.0)
        self.declare_parameter("synchronization_cache_size", 8)
        self.declare_parameter("synchronization_timeout_sec", 1.5)
        self.declare_parameter("mask_alpha", 0.4)
        self.declare_parameter("display_max_fps", 30.0)
        self.declare_parameter("show_mask", True)
        self.declare_parameter("show_boxes", True)
        image_topic = self.get_parameter("image_topic").value
        result_topic = self.get_parameter("result_topic").value
        action_name = self.get_parameter("qwen_action_name").value
        self.create_subscription(CompressedImage, image_topic, self._on_image, SENSOR_QOS)
        result_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=4,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(ArgusInferenceResult, result_topic, self._on_result, result_qos)
        self._action = ActionClient(self, QwenInference, action_name)
        self._commands: deque[tuple] = deque(maxlen=8)
        self._commands_lock = threading.Lock()
        self._goal_handle = None
        self._last_image = self._last_result = 0.0
        self._image_times: deque[float] = deque(maxlen=60)
        self._result_times: deque[float] = deque(maxlen=60)
        self.create_timer(0.2, self._poll)

    def queue_goal(self, prompt: str, include_context: bool) -> None:
        with self._commands_lock:
            self._commands.append(("goal", prompt, include_context))

    def queue_cancel(self) -> None:
        with self._commands_lock:
            self._commands.append(("cancel",))

    def _on_image(self, message: CompressedImage) -> None:
        now = time.monotonic()
        self._last_image = now
        self._image_times.append(now)
        self._processor.submit_image(stamp_to_nanoseconds(message.header.stamp),
                                     bytes(message.data), now)

    def _on_result(self, message: ArgusInferenceResult) -> None:
        now = time.monotonic()
        self._last_result = now
        self._result_times.append(now)
        self._processor.submit_result(stamp_to_nanoseconds(message.header.stamp), message)

    @staticmethod
    def _fps(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        return (len(times) - 1) / duration if duration > 0 else 0.0

    def _poll(self) -> None:
        now = time.monotonic()
        self._bridge.connection.emit({
            "image_online": bool(self._last_image and now - self._last_image <
                                 float(self.get_parameter("image_timeout_sec").value)),
            "result_online": bool(self._last_result and now - self._last_result < 3.0),
            "action_online": self._action.server_is_ready(),
            "image_fps": self._fps(self._image_times), "result_fps": self._fps(self._result_times),
        })
        with self._commands_lock:
            commands = list(self._commands)
            self._commands.clear()
        for command in commands:
            if command[0] == "goal":
                self._send_goal(command[1], command[2])
            elif command[0] == "cancel":
                self._cancel_goal()

    def _send_goal(self, prompt: str, include_context: bool) -> None:
        if self._goal_handle is not None:
            self._bridge.qwen_error.emit("已有 Qwen 请求正在执行")
            return
        if not self._action.server_is_ready():
            self._bridge.qwen_error.emit("Qwen Action Server 不可用")
            return
        goal = QwenInference.Goal()
        goal.prompt, goal.include_detection_context = prompt, include_context
        self._bridge.qwen_state.emit("正在提交")
        future = self._action.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._goal_response)

    def _goal_response(self, future: Any) -> None:
        try:
            handle = future.result()
            if not handle.accepted:
                self._bridge.qwen_error.emit("Qwen 请求被服务端拒绝")
                return
            self._goal_handle = handle
            self._bridge.qwen_state.emit("推理中")
            result_future = handle.get_result_async()
            result_future.add_done_callback(self._result_response)
        except Exception as exc:
            self._goal_handle = None
            self._bridge.qwen_error.emit(f"提交 Qwen 请求失败: {exc}")

    def _on_feedback(self, message: Any) -> None:
        self._bridge.qwen_state.emit(message.feedback.stage or "推理中")

    def _result_response(self, future: Any) -> None:
        try:
            wrapped = future.result()
            self._bridge.qwen_result.emit(wrapped.result.result)
        except Exception as exc:
            self._bridge.qwen_error.emit(f"获取 Qwen 结果失败: {exc}")
        finally:
            self._goal_handle = None

    def _cancel_goal(self) -> None:
        if self._goal_handle is None:
            self._bridge.qwen_error.emit("当前没有可取消的 Qwen 请求")
            return
        self._bridge.qwen_state.emit("正在请求取消")
        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(self._cancel_response)

    def _cancel_response(self, future: Any) -> None:
        try:
            response = future.result()
            state = "取消请求已接受，等待服务端结束" if response.goals_canceling else "服务端未接受取消请求"
            self._bridge.qwen_state.emit(state)
        except Exception as exc:
            self._bridge.qwen_error.emit(f"取消请求失败: {exc}")


class RosWorker:
    def __init__(self, bridge: Any, processor: Any, ros_args: list[str]) -> None:
        self._bridge, self._processor, self._ros_args = bridge, processor, ros_args
        self._thread: threading.Thread | None = None
        self._executor: SingleThreadedExecutor | None = None
        self.node: OperatorNode | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="argus-ros", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        try:
            rclpy.init(args=self._ros_args)
            self.node = OperatorNode(self._bridge, self._processor)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self.node)
            self._bridge.ros_ready.emit(self.node)
            self._executor.spin()
        except Exception as exc:
            self._bridge.fatal_error.emit(f"ROS2 worker 启动失败: {exc}")
        finally:
            if self.node:
                self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    def stop(self) -> None:
        if self.node:
            self.node.queue_cancel()
        if self._executor:
            self._executor.shutdown(timeout_sec=1.0)
        if self._thread:
            self._thread.join(timeout=3.0)
