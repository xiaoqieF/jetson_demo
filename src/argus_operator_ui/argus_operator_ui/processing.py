from collections import deque
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage

from .overlay import draw_overlay
from .synchronizer import FrameSynchronizer


class ImageProcessor:
    """Decode and overlay thread with one replaceable pending image."""

    def __init__(self, frame_callback: Callable[[np.ndarray, dict], None],
                 error_callback: Callable[[str], None], cache_size: int,
                 timeout_sec: float, mask_alpha: float) -> None:
        self._frame_callback = frame_callback
        self._error_callback = error_callback
        self._sync = FrameSynchronizer(cache_size, timeout_sec)
        self._mask_alpha = mask_alpha
        self._show_mask = True
        self._show_boxes = True
        self._condition = threading.Condition()
        self._pending_image: Optional[tuple] = None
        self._pending_results: deque[tuple] = deque(maxlen=cache_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="argus-image", daemon=False)
        self._thread.start()

    def submit_image(self, stamp_ns: int, jpeg: bytes, received_at: float) -> None:
        with self._condition:
            self._pending_image = (stamp_ns, jpeg, received_at)
            self._condition.notify()

    def submit_result(self, stamp_ns: int, result: Any) -> None:
        with self._condition:
            self._pending_results.append((stamp_ns, result))
            self._condition.notify()

    def set_layers(self, show_mask: bool, show_boxes: bool) -> None:
        with self._condition:
            self._show_mask, self._show_boxes = show_mask, show_boxes

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: not self._running or self._pending_results or self._pending_image)
                if not self._running:
                    return
                results = list(self._pending_results)
                self._pending_results.clear()
                image_task, self._pending_image = self._pending_image, None
                show_mask, show_boxes = self._show_mask, self._show_boxes
            for stamp_ns, result in results:
                match = self._sync.insert_result(stamp_ns, result)
                if match:
                    self._emit_overlay(match.image, result, show_mask, show_boxes)
            if image_task:
                stamp_ns, jpeg, received_at = image_task
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    self._error_callback("JPEG 解码失败")
                    continue
                self._emit_frame(frame, {"stamp_ns": stamp_ns,
                    "image_latency_ms": (time.monotonic() - received_at) * 1000.0})
                match = self._sync.insert_image(stamp_ns, frame)
                if match:
                    self._emit_overlay(frame, match.result, show_mask, show_boxes)

    def _emit_overlay(self, frame: np.ndarray, result: Any,
                      show_mask: bool, show_boxes: bool) -> None:
        try:
            output = draw_overlay(frame, result, show_mask, show_boxes, self._mask_alpha)
            self._emit_frame(output, {"stamp_ns": 0, "frame_number": int(result.frame_number),
                "target_count": len(result.instances), "inference_ms": float(result.inference_ms)})
        except Exception as exc:
            self._error_callback(f"绘制检测结果失败: {exc}")

    def _emit_frame(self, frame: np.ndarray, stats: dict) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                       QImage.Format_RGB888).copy()
        self._frame_callback(image, stats)
