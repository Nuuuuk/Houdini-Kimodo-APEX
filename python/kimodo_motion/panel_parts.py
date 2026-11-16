"""pieces of the panel ui."""

import time

import hou
from PySide6 import QtCore, QtGui, QtWidgets

from . import doc, motion_node

TRACKS = ("Prompts", "Full-Body", "2D Root", "Left Hand", "Right Hand", "Left Foot", "Right Foot")
WIDE_ASPECT = 2.2

STAGE_COLORS = {
    "idle": "#8a8a8a",
    "starting": "#d9a441",
    "loading": "#d9a441",
    "generating": "#4a90d9",
    "canceling": "#d9a441",
    "done": "#5cb85c",
    "canceled": "#8a8a8a",
    "error": "#d9534f",
}


class Metrics:
    def __init__(self, widget):
        fm = widget.fontMetrics()
        self.row_h = fm.height() + 8
        self.ruler_h = fm.height() + 6
        self.label_w = max(fm.horizontalAdvance(name) for name in TRACKS) + 14
        self.px_per_frame = max(3, round(fm.height() / 5))
        self.tracks_h = self.ruler_h + self.row_h * len(TRACKS)
        self.wide_min_width = 60 * fm.height()


class FontSized(QtWidgets.QWidget):
    """redoes its metrics when the font changes, houdini styles panels late"""

    def __init__(self):
        super().__init__()
        self.apply_metrics()

    def apply_metrics(self):
        self.m = Metrics(self)

    def changeEvent(self, event):
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self.apply_metrics()
            self.update()
        super().changeEvent(event)


class TrackLabels(FontSized):
    def apply_metrics(self):
        super().apply_metrics()
        self.setFixedSize(self.m.label_w, self.m.tracks_h)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setPen(self.palette().color(QtGui.QPalette.Text))
        for i, name in enumerate(TRACKS):
            p.drawText(
                QtCore.QRect(6, self.m.ruler_h + i * self.m.row_h, self.m.label_w - 8, self.m.row_h),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                name,
            )
        p.end()


class TimelineCanvas(FontSized):
    def __init__(self):
        self.prompts, self.total, self.frame = [], 5 * doc.FPS, 0
        super().__init__()

    def apply_metrics(self):
        super().apply_metrics()
        self.setFixedHeight(self.m.tracks_h)
        self.setMinimumWidth(self.total * self.m.px_per_frame)

    def set_state(self, prompts, frame):
        total = prompts[-1]["end_frame"] if prompts else 5 * doc.FPS
        if (prompts, frame, total) != (self.prompts, self.frame, self.total):
            self.prompts, self.frame, self.total = prompts, frame, total
            self.setMinimumWidth(total * self.m.px_per_frame)
            self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        pal = self.palette()
        ruler_h, row_h = self.m.ruler_h, self.m.row_h
        scale = max(self.width() / max(self.total, 1), self.m.px_per_frame)
        text = pal.color(QtGui.QPalette.Text)
        for i in range(len(TRACKS)):
            role = QtGui.QPalette.AlternateBase if i % 2 else QtGui.QPalette.Base
            p.fillRect(0, ruler_h + i * row_h, self.width(), row_h - 1, pal.color(role))
        p.setPen(text)
        for second in range(self.total // doc.FPS + 1):
            x = int(second * doc.FPS * scale)
            p.drawLine(x, ruler_h - 6, x, ruler_h - 1)
            p.drawText(x + 3, ruler_h - 6, f"{second}s")
        for pr in self.prompts:
            x0, x1 = int(pr["start_frame"] * scale), int(pr["end_frame"] * scale)
            rect = QtCore.QRect(x0 + 1, ruler_h + 2, x1 - x0 - 2, row_h - 5)
            p.fillRect(rect, QtGui.QColor(70, 120, 200))
            p.setPen(QtGui.QColor(255, 255, 255))
            label = p.fontMetrics().elidedText(pr["text"], QtCore.Qt.ElideRight, rect.width() - 8)
            p.drawText(rect.adjusted(4, 0, -4, 0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label)
        x = int(self.frame * scale)
        p.setPen(QtGui.QPen(QtGui.QColor(230, 90, 60), 2))
        p.drawLine(x, 0, x, self.height())
        p.end()


class TimelineArea(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.canvas = TimelineCanvas()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.labels, self.scroll = TrackLabels(), scroll
        row.addWidget(self.labels, 0, QtCore.Qt.AlignTop)
        row.addWidget(scroll, 1)

    def minimumSizeHint(self):
        bar = self.scroll.horizontalScrollBar().sizeHint().height()
        return QtCore.QSize(self.labels.width() + 100, self.labels.height() + bar + 2)

    def sizeHint(self):
        return self.minimumSizeHint()


class StatusWidget(QtWidgets.QFrame):
    dismissed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(8, 6, 8, 6)
        col.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        self.dot = QtWidgets.QLabel()
        self.dot.setFixedSize(10, 10)
        self.text = QtWidgets.QLabel("Press Generate.")
        self.text.setWordWrap(True)
        self.elapsed = QtWidgets.QLabel("")
        head.addWidget(self.dot, 0, QtCore.Qt.AlignVCenter)
        head.addWidget(self.text, 1)
        head.addWidget(self.elapsed, 0, QtCore.Qt.AlignTop)
        col.addLayout(head)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        col.addWidget(self.bar)
        self.error = QtWidgets.QWidget()
        err = QtWidgets.QVBoxLayout(self.error)
        err.setContentsMargins(0, 2, 0, 0)
        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.detail.setStyleSheet(f"color: {STAGE_COLORS['error']};")
        actions = QtWidgets.QHBoxLayout()
        self.log_link = QtWidgets.QLabel()
        self.log_link.setOpenExternalLinks(True)
        dismiss = QtWidgets.QPushButton("Dismiss")
        dismiss.setToolTip("Clear the warning or error on the node.")
        dismiss.clicked.connect(self.dismissed)
        actions.addWidget(self.log_link, 1)
        actions.addWidget(dismiss, 0)
        err.addWidget(self.detail)
        err.addLayout(actions)
        col.addWidget(self.error)
        self.show_status(
            {
                "stage": "idle",
                "text": "Press Generate.",
                "progress": None,
                "started": None,
                "elapsed": None,
                "detail": "",
                "log": "",
            }
        )

    def show_status(self, st):
        stage = st["stage"]
        self.dot.setStyleSheet(f"background: {STAGE_COLORS.get(stage, '#8a8a8a')}; border-radius: 5px;")
        self.text.setText(st["text"])
        seconds = st["elapsed"]
        if stage in motion_node.ACTIVE_STAGES and st["started"] is not None:
            seconds = time.monotonic() - st["started"]
        self.elapsed.setText("" if seconds is None else f"{seconds:.0f} s")
        active = stage in motion_node.ACTIVE_STAGES
        self.bar.setVisible(active)
        if active:
            if st["progress"] is None:
                self.bar.setRange(0, 0)
            else:
                self.bar.setRange(0, 100)
                self.bar.setValue(int(st["progress"] * 100))
        self.error.setVisible(stage == "error")
        if stage == "error":
            self.detail.setText(st["detail"])
            url = QtCore.QUrl.fromLocalFile(st["log"]).toString()
            self.log_link.setText(f'<a href="{url}">Open service.log</a>' if st["log"] else "")


class KimodoPanel(QtWidgets.QWidget):
    def __init__(self, parm_editor_factory=None):
        super().__init__()
        self.node = None
        self.parm_editor = None
        self.parm_editor_factory = parm_editor_factory or (lambda node: hou.qt.ParmDialog(node, compact=True))
        self.timeline = TimelineArea()
        self.status = StatusWidget()
        self.status.dismissed.connect(self.dismiss_error)
        self.lower = QtWidgets.QWidget()
        self.lower_layout = QtWidgets.QVBoxLayout(self.lower)
        self.lower_layout.setContentsMargins(0, 0, 0, 0)
        self.lower_layout.setSpacing(4)
        self.lower_layout.addWidget(self.status, 0)
        self.lower_layout.addStretch(1)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.timeline)
        self.splitter.addWidget(self.lower)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.splitter)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def set_node(self, node):
        self.node = node
        if node is not None:
            if self.parm_editor is None:
                self.parm_editor = self.parm_editor_factory(node)
                self.lower_layout.takeAt(1)
                self.lower_layout.addWidget(self.parm_editor, 1)
            else:
                self.parm_editor.setNode(node)
        self.refresh()

    def refresh(self):
        """cheap, runs five times a second"""
        try:
            if self.node is None:
                return
            self.status.show_status(motion_node.get_status(self.node))
            prompts = motion_node.read_doc(self.node)["prompts"]
            self.timeline.canvas.set_state(prompts, int(round(hou.frame())) - 1)
        except (hou.ObjectWasDeleted, doc.DocError):
            self.node = None

    def dismiss_error(self):
        if self.node is not None:
            motion_node.clear_error(self.node)
            self.refresh()

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        wide = w >= Metrics(self).wide_min_width and w > WIDE_ASPECT * h
        orientation = QtCore.Qt.Horizontal if wide else QtCore.Qt.Vertical
        if orientation != self.splitter.orientation():
            self.splitter.setOrientation(orientation)
            if wide:
                self.splitter.setSizes([w * 3 // 5, w * 2 // 5])
            else:
                top = self.timeline.minimumSizeHint().height()
                self.splitter.setSizes([top, max(h - top, 1)])
        super().resizeEvent(event)

    def close_panel(self):
        self.timer.stop()
        self.node = None
