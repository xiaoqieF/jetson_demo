import argparse
import signal
import sys

from PySide6.QtWidgets import QApplication

from .processing import ImageProcessor
from .ui import EventBridge, MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus ROS2 operator UI")
    parser.add_argument("--demo", action="store_true", help="run without ROS2 using synthetic frames")
    parser.add_argument("--demo-video", help="local video path used by demo mode")
    known, ros_args = parser.parse_known_args()
    app = QApplication(sys.argv[:1])
    bridge = EventBridge()
    processor = ImageProcessor(lambda image, stats: bridge.frame.emit(image, stats),
                               bridge.qwen_error.emit, 8, 1.5, 0.4)
    ros_worker = demo_source = None
    if known.demo:
        from .demo import DemoSource
        demo_source = DemoSource(processor, known.demo_video)
    else:
        from .ros_worker import RosWorker
        ros_worker = RosWorker(bridge, processor, ros_args)

    stopped = False
    def shutdown() -> None:
        nonlocal stopped
        if stopped:
            return
        stopped = True
        if demo_source:
            demo_source.stop()
        if ros_worker:
            ros_worker.stop()
        processor.stop()

    window = MainWindow(bridge, processor, shutdown)
    window.show()
    processor.start()
    if demo_source:
        demo_source.start()
        bridge.connection.emit({"image_online": True, "result_online": True,
                                "action_online": False, "image_fps": 20.0, "result_fps": 20.0})
    if ros_worker:
        ros_worker.start()
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.aboutToQuit.connect(shutdown)
    exit_code = app.exec()
    shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
