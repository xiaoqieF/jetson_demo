import html
import time
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFontDatabase, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
                               QStatusBar, QTextBrowser, QVBoxLayout, QWidget)


class EventBridge(QObject):
    frame = Signal(object, object)
    connection = Signal(object)
    qwen_state = Signal(str)
    qwen_result = Signal(object)
    qwen_error = Signal(str)
    fatal_error = Signal(str)
    ros_ready = Signal(object)


class VideoLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("等待图像流…")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background: #15181c; color: #9aa4ad;")
        self._image: QImage | None = None

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._refresh()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._image:
            pixmap = QPixmap.fromImage(self._image).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, bridge: EventBridge, processor: Any, shutdown_callback: Any) -> None:
        super().__init__()
        self._bridge, self._processor = bridge, processor
        self._shutdown_callback = shutdown_callback
        self._node = None
        self._busy = False
        self._last_frame_time = 0.0
        self._messages: list[tuple[str, str, str]] = []
        self._pending_prompt = ""
        self.setWindowTitle("Argus Operator — ROS2 / Qwen")
        self.resize(1280, 760)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #11151a; color: #dce3ea; }
            QFrame#chatPanel { background: #181d24; border-left: 1px solid #29313b; }
            QLabel#panelTitle { color: #f3f6f8; font-size: 18px; font-weight: 650; }
            QLabel#panelSubtitle { color: #7f8b98; font-size: 12px; }
            QLabel#statePill { background: #232b35; color: #aeb9c5; border-radius: 10px;
                               padding: 4px 10px; }
            QLabel#errorLabel { color: #f08b8b; padding: 2px 4px; }
            QTextBrowser { background: #13181e; border: 1px solid #29313b;
                           border-radius: 10px; padding: 8px; selection-background-color: #35699b; }
            QPlainTextEdit { background: #20262e; border: 1px solid #333d48; border-radius: 9px;
                             padding: 9px; color: #eef2f5; selection-background-color: #35699b; }
            QPlainTextEdit:focus { border: 1px solid #4f91cf; }
            QPushButton { background: #2f78b7; border: 0; border-radius: 8px; padding: 8px 17px;
                          color: white; font-weight: 600; }
            QPushButton:hover { background: #3b88c8; }
            QPushButton:disabled { background: #303943; color: #71808d; }
            QPushButton#cancelButton { background: #353e48; }
            QPushButton#cancelButton:hover { background: #48535f; }
            QCheckBox { color: #aeb8c2; spacing: 6px; }
            QStatusBar { background: #0d1116; color: #98a5b2; border-top: 1px solid #252d35; }
            QSplitter::handle { background: #29313b; width: 1px; }
        """)
        self.video = VideoLabel()
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("输入消息，询问当前画面…")
        self.prompt.setFixedHeight(82)
        self.include_context = QCheckBox("附带 YOLO 检测上下文")
        self.include_context.setChecked(True)
        self.show_mask = QCheckBox("显示 Mask")
        self.show_mask.setChecked(True)
        self.show_boxes = QCheckBox("显示检测框")
        self.show_boxes.setChecked(True)
        self.send = QPushButton("发送")
        self.send.setEnabled(False)
        self.cancel = QPushButton("取消")
        self.cancel.setObjectName("cancelButton")
        self.cancel.setEnabled(False)
        self.state = QLabel("等待 ROS2…")
        self.state.setObjectName("statePill")
        self.state.setAlignment(Qt.AlignCenter)
        self.elapsed = QLabel("—")
        self.error = QLabel("")
        self.error.setObjectName("errorLabel")
        self.error.setWordWrap(True)
        self.error.hide()
        self.response = QTextBrowser()
        self.response.setReadOnly(True)
        self.response.setOpenExternalLinks(False)
        self.response.setPlaceholderText("对话将显示在这里")
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.include_context)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.send)
        layers = QHBoxLayout()
        layers.setContentsMargins(0, 0, 0, 0)
        layers.addWidget(self.show_boxes)
        layers.addWidget(self.show_mask)
        layers.addStretch(1)
        layers.addWidget(self.state)
        title = QLabel("Qwen 视觉助手")
        title.setObjectName("panelTitle")
        subtitle = QLabel("围绕当前相机画面进行对话")
        subtitle.setObjectName("panelSubtitle")
        header_text = QVBoxLayout()
        header_text.setSpacing(1)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header = QHBoxLayout()
        header.addLayout(header_text)
        header.addStretch(1)
        header.addWidget(QLabel("耗时"))
        header.addWidget(self.elapsed)
        right = QFrame()
        right.setObjectName("chatPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(10)
        right_layout.addLayout(header)
        right_layout.addLayout(layers)
        right_layout.addWidget(self.response, 1)
        right_layout.addWidget(self.error)
        right_layout.addWidget(self.prompt)
        right_layout.addLayout(buttons)
        splitter = QSplitter()
        splitter.addWidget(self.video)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.status = QLabel("图像: 离线 | YOLO: 未连接 | Qwen: 未连接")
        self.metrics = QLabel("图像 0.0 FPS | YOLO 0.0 FPS | 延迟 — | 推理 — | 帧 — | 目标 —")
        self.metrics.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.metrics.setFixedWidth(620)
        self.metrics.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.metrics)
        self.send.clicked.connect(self._send)
        self.cancel.clicked.connect(self._cancel)
        self.show_mask.toggled.connect(self._layers_changed)
        self.show_boxes.toggled.connect(self._layers_changed)
        bridge.frame.connect(self._on_frame)
        bridge.connection.connect(self._on_connection)
        bridge.qwen_state.connect(self._on_qwen_state)
        bridge.qwen_result.connect(self._on_qwen_result)
        bridge.qwen_error.connect(self._on_qwen_error)
        bridge.fatal_error.connect(self._fatal)
        bridge.ros_ready.connect(self._ros_ready)

    def _ros_ready(self, node: Any) -> None:
        self._node = node
        self.show_mask.setChecked(bool(node.get_parameter("show_mask").value))
        self.show_boxes.setChecked(bool(node.get_parameter("show_boxes").value))

    def _layers_changed(self) -> None:
        self._processor.set_layers(self.show_mask.isChecked(), self.show_boxes.isChecked())

    def _send(self) -> None:
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            self._set_error("Prompt 不能为空")
            return
        if not self._node:
            self._set_error("ROS2 尚未就绪")
            return
        self._busy = True
        self._pending_prompt = prompt
        self._append_message("user", "你", prompt)
        self.prompt.clear()
        self.send.setEnabled(False)
        self.cancel.setEnabled(True)
        self._set_error("")
        self.state.setText("排队发送")
        self._node.queue_goal(prompt, self.include_context.isChecked())

    def _cancel(self) -> None:
        if self._node:
            self.state.setText("正在请求取消")
            self._node.queue_cancel()

    def _on_frame(self, image: QImage, stats: dict) -> None:
        self.video.set_image(image)
        latency = stats.get("image_latency_ms")
        frame_number = stats.get("frame_number", "—")
        count = stats.get("target_count", "—")
        inference = stats.get("inference_ms")
        self._last_frame_time = time.monotonic()
        old = self.metrics.property("connection") or {}
        old.update({"latency": latency, "frame": frame_number, "count": count,
                    "inference": inference})
        self.metrics.setProperty("connection", old)
        self._render_metrics(old)

    def _on_connection(self, info: dict) -> None:
        self.status.setText(f"图像: {'在线' if info['image_online'] else '流断开'} | "
                            f"YOLO: {'在线' if info['result_online'] else '未收到'} | "
                            f"Qwen: {'可用' if info['action_online'] else '不可用'}")
        self.send.setEnabled(info["action_online"] and not self._busy)
        old = self.metrics.property("connection") or {}
        old.update(info)
        self.metrics.setProperty("connection", old)
        self._render_metrics(old)

    def _render_metrics(self, info: dict) -> None:
        latency = "   —   " if info.get("latency") is None else f"{info['latency']:6.0f}ms"
        inference = "   —   " if info.get("inference") is None else f"{info['inference']:6.1f}ms"
        frame = str(info.get("frame", "—"))[-8:].rjust(8)
        count = str(info.get("count", "—"))[-3:].rjust(3)
        self.metrics.setText(f"IMG {info.get('image_fps', 0):5.1f}fps  "
            f"YOLO {info.get('result_fps', 0):5.1f}fps  LAT {latency}  "
            f"INF {inference}  FRAME {frame}  OBJ {count}")

    def _on_qwen_state(self, state: str) -> None:
        self.state.setText(state)

    def _on_qwen_result(self, result: Any) -> None:
        self._busy = False
        self.send.setEnabled(True)
        self.cancel.setEnabled(False)
        self.state.setText("完成" if result.success else "失败")
        if result.description:
            self._append_message("assistant", "Qwen", result.description)
        elif not result.success:
            self._append_message("error", "系统", result.error_message or "推理失败")
        self.elapsed.setText(f"{float(result.inference_ms):.1f} ms")
        self._set_error(result.error_message)
        self._pending_prompt = ""

    def _on_qwen_error(self, message: str) -> None:
        self._busy = False
        self.send.setEnabled(True)
        self.cancel.setEnabled(False)
        self.state.setText("错误")
        self._set_error(message)
        self._append_message("error", "系统", message)
        self._pending_prompt = ""

    def _append_message(self, role: str, sender: str, text: str) -> None:
        self._messages.append((role, sender, text))
        self._messages = self._messages[-30:]
        blocks = []
        for message_role, message_sender, message_text in self._messages:
            safe_sender = html.escape(message_sender)
            safe_text = html.escape(message_text).replace("\n", "<br>")
            if message_role == "user":
                align, background, foreground = "right", "#285f8f", "#ffffff"
            elif message_role == "error":
                align, background, foreground = "left", "#5a3034", "#ffd9dc"
            else:
                align, background, foreground = "left", "#28313b", "#e8edf2"
            blocks.append(
                f'<div style="text-align:{align}; margin:8px 2px;">'
                f'<span style="color:#8794a1; font-size:11px;">{safe_sender}</span><br>'
                f'<span style="background:{background}; color:{foreground}; border-radius:8px; '
                f'padding:7px 10px; line-height:1.45;">{safe_text}</span></div>')
        self.response.setHtml("".join(blocks))
        self.response.moveCursor(QTextCursor.End)

    def _set_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.setVisible(bool(message))

    def _fatal(self, message: str) -> None:
        self._set_error(message)
        QMessageBox.critical(self, "Argus Operator", message)

    def closeEvent(self, event: Any) -> None:
        self.setEnabled(False)
        self._shutdown_callback()
        event.accept()
