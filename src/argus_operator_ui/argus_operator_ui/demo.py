from types import SimpleNamespace
import threading
import time

import cv2
import numpy as np


class DemoSource:
    def __init__(self, processor, video_path: str | None = None) -> None:
        self._processor = processor
        self._video_path = video_path
        self._running = False
        self._thread = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="argus-demo", daemon=False)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        capture = cv2.VideoCapture(self._video_path) if self._video_path else None
        frame_number = 0
        while self._running:
            if capture:
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            else:
                frame = np.full((640, 960, 3), (28, 31, 36), np.uint8)
                cv2.putText(frame, "Argus Operator Demo", (230, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 220, 220), 2, cv2.LINE_AA)
            stamp_ns = time.time_ns()
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self._processor.submit_image(stamp_ns, jpeg.tobytes(), time.monotonic())
                x = 80 + (frame_number * 7) % max(1, frame.shape[1] - 320)
                mask = np.zeros((120, 180), np.uint8)
                cv2.ellipse(mask, (90, 60), (75, 52), 0, 0, 360, 255, -1)
                item = SimpleNamespace(class_id=0, class_name="demo_object", confidence=0.92,
                    x_min=float(x), y_min=180.0, x_max=float(x + 180), y_max=300.0,
                    mask_x=x, mask_y=180, mask_width=180, mask_height=120,
                    mask=mask.reshape(-1))
                result = SimpleNamespace(image_width=frame.shape[1], image_height=frame.shape[0],
                    inference_ms=12.4, frame_number=frame_number, instances=[item])
                self._processor.submit_result(stamp_ns, result)
            frame_number += 1
            time.sleep(0.05)
        if capture:
            capture.release()
