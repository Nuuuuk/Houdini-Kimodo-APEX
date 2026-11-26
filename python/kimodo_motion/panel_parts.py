"""pieces of the panel ui."""

import time

import hou
from PySide6 import QtCore, QtGui, QtWidgets

from . import doc, motion_node, timeline, timeline_edit

TRACKS = ("Prompts", "Full-Body", "2D Root", "Left Hand", "Right Hand", "Left Foot", "Right Foot")
TRACK_TYPES = ("fullbody", "root2d", "left-hand", "right-hand", "left-foot", "right-foot")

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
HINT_COLOR = "#a3a3a3"


def paste_entries(ctype, frame):
    """[(text, keep the frame's root)] for a track's paste entries"""
    n = frame + 1
    if ctype == "fullbody":
        return [(f"Paste Pose and Root at Frame {n}", False), (f"Paste Pose Only at Frame {n}", True)]
    if ctype == "root2d":
        return [(f"Paste Root at Frame {n}", False)]
    return [(f"Paste Pose at Frame {n}", False)]


class Look:
    def __init__(self, palette):
        base = palette.color(QtGui.QPalette.Base)
        self.text = palette.color(QtGui.QPalette.Text)
        self.dim = QtGui.QColor(self.text)
        self.dim.setAlpha(150)
        self.row, self.row_alt = base.darker(140), base.darker(127)
        self.ruler, self.header = base.darker(170), base.darker(112)
        self.line = base.darker(240)
        self.grid = QtGui.QColor(255, 255, 255, 13)
        self.beyond = QtGui.QColor(0, 0, 0, 95)


class Metrics:
    def __init__(self, widget):
        fm = widget.fontMetrics()
        self.row_h = fm.height() + 8
        self.ruler_h = fm.height() + 6
        self.label_w = max(fm.horizontalAdvance(name) for name in TRACKS) + 14
        self.px_per_frame = max(3, round(fm.height() / 5))
        self.tracks_h = self.ruler_h + self.row_h * len(TRACKS)


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
        look = Look(self.palette())
        width, ruler_h, row_h = self.width() - 1, self.m.ruler_h, self.m.row_h
        p.fillRect(self.rect(), look.line)
        p.fillRect(0, 0, width, ruler_h - 1, look.ruler)
        p.setPen(look.text)
        for i, name in enumerate(TRACKS):
            p.fillRect(0, ruler_h + i * row_h, width, row_h - 1, look.header)
            p.drawText(
                QtCore.QRect(6, ruler_h + i * row_h, self.m.label_w - 8, row_h - 1),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                name,
            )
        p.end()


class TimelineArea(QtWidgets.QWidget):
    def __init__(self, canvas):
        super().__init__()
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.canvas = canvas
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.labels, self.scroll = TrackLabels(), scroll
        scroll.horizontalScrollBar().valueChanged.connect(lambda _: self.canvas.update())
        row.addWidget(self.labels, 0, QtCore.Qt.AlignTop)
        row.addWidget(scroll, 1)

    def minimumSizeHint(self):
        bar = self.scroll.horizontalScrollBar().sizeHint().height()
        return QtCore.QSize(self.labels.width() + 100, self.labels.height() + bar + 2)

    def sizeHint(self):
        return self.minimumSizeHint()


class HintLabel(QtWidgets.QLabel):
    """always two lines high so the panel doesn't jump"""

    def __init__(self):
        super().__init__("")
        self.setWordWrap(True)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.fit()

    def fit(self):
        self.setMinimumHeight(2 * self.fontMetrics().lineSpacing())

    def changeEvent(self, event):
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self.fit()
        super().changeEvent(event)


class StatusWidget(QtWidgets.QFrame):
    dismissed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
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
        self.hint = HintLabel()
        self.hint.setStyleSheet(f"color: {HINT_COLOR};")
        col.addWidget(self.hint)
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
        dismiss.clicked.connect(self.dismissed)
        actions.addWidget(self.log_link, 1)
        actions.addWidget(dismiss, 0)
        err.addWidget(self.detail)
        err.addLayout(actions)
        col.addWidget(self.error)
        self.shown = None
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

    def show_hint(self, text):
        if text != self.hint.text():
            self.hint.setText(text)

    def show_status(self, st):
        stage = st["stage"]
        seconds = st["elapsed"]
        if stage in motion_node.ACTIVE_STAGES and st["started"] is not None:
            seconds = time.monotonic() - st["started"]
        elapsed = "" if seconds is None else f"{seconds:.0f} s"
        shown = (stage, st["text"], st["progress"], st["detail"], st["log"], elapsed)
        if shown == self.shown:
            return
        self.shown = shown
        self.dot.setStyleSheet(f"background: {STAGE_COLORS.get(stage, '#8a8a8a')}; border-radius: 5px;")
        self.text.setText(st["text"])
        self.elapsed.setText(elapsed)
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


MENU_RIGHT = 310
DIVIDER_SPACE = 8
GENERATE_TALLER = 1.3
ACCENT_RGB = (70, 120, 200)
GENERATE_STYLE = (
    "QPushButton {{ background-color: {normal}; color: white; border: 1px solid {edge}; border-radius: {radius}px; }}"
    " QPushButton:hover {{ background-color: {hover}; }} QPushButton:pressed {{ background-color: {down}; }}"
)


def ui_scale():
    return hou.ui.globalScaleFactor() if hou.isUIAvailable() else 1.0


class ClipFields(QtWidgets.QWidget):
    LABEL_X = 6

    def __init__(self):
        super().__init__()
        self.node = None
        self.run = None
        self.preview = None
        self.prompt_label, self.duration_label = QtWidgets.QLabel("Prompt"), QtWidgets.QLabel("Duration (s)")
        self.prompt, self.duration = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(doc.MIN_PROMPT_FRAMES, doc.MAX_PROMPT_FRAMES)
        rows = QtWidgets.QVBoxLayout(self)
        rows.setContentsMargins(self.LABEL_X, 0, self.LABEL_X, 0)
        rows.setSpacing(4)
        for widgets in ((self.prompt_label, self.prompt), (self.duration_label, self.duration, self.slider)):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for w in widgets:
                row.addWidget(w, 1 if w in (self.prompt, self.slider) else 0)
            rows.addLayout(row)
        self.prompt.editingFinished.connect(self.commit_prompt)
        self.duration.editingFinished.connect(self.commit_duration)
        for field in (self.prompt, self.duration):
            field.installEventFilter(self)
        self.slider.valueChanged.connect(self.slid)
        self.slider.sliderReleased.connect(self.slide_ended)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    # a click anywhere else commits the field first, so generate gets what was typed
    def eventFilter(self, obj, event):
        kind = event.type()
        if obj in (self.prompt, self.duration) and kind in (QtCore.QEvent.FocusIn, QtCore.QEvent.FocusOut):
            app = QtWidgets.QApplication.instance()
            app.installEventFilter(self) if kind == QtCore.QEvent.FocusIn else app.removeEventFilter(self)
        elif kind == QtCore.QEvent.MouseButtonPress and isinstance(obj, QtWidgets.QWidget):
            for field, commit in ((self.prompt, self.commit_prompt), (self.duration, self.commit_duration)):
                if field.hasFocus() and obj is not field and not field.isAncestorOf(obj):
                    commit()
                    field.clearFocus()
        return super().eventFilter(obj, event)

    def place(self, left):
        label = max(left - self.LABEL_X - 4, 10)
        number = max(round(self.duration.fontMetrics().horizontalAdvance("00.000") * 1.6), 40)
        for w, size in ((self.prompt_label, label), (self.duration_label, label), (self.duration, number)):
            if w.width() != size:
                w.setFixedWidth(size)

    def clip(self):
        if self.node is None:
            return None
        d = motion_node.read_doc(self.node)
        return d["prompts"][motion_node.selected_prompt(self.node, d)] if d["prompts"] else None

    def refresh(self, node):
        """show the selected clip unless a field is being typed in or dragged"""
        self.node = node
        clip = self.clip()
        self.setEnabled(clip is not None)
        if clip is None:
            return
        text, frames = clip["text"], clip["end_frame"] - clip["start_frame"]
        if not self.prompt.isModified() and self.prompt.text() != text:
            self.prompt.setText(text)
            self.prompt.setCursorPosition(0)
        if (
            not self.duration.isModified()
            and not self.slider.isSliderDown()
            and self.duration.text() != f"{frames / doc.FPS:g}"
        ):
            self.duration.setText(f"{frames / doc.FPS:g}")
        if not self.slider.isSliderDown() and self.slider.value() != frames:
            self.slider.blockSignals(True)
            self.slider.setValue(frames)
            self.slider.blockSignals(False)

    def commit_prompt(self):
        self.prompt.setModified(False)
        clip, text = self.clip(), self.prompt.text().strip()
        if clip is not None and text and text != clip["text"] and self.run is not None:
            self.run(timeline_edit.set_prompt_text, clip["id"], text)

    def commit_duration(self):
        self.duration.setModified(False)
        clip = self.clip()
        if clip is None:
            return
        try:
            frames = timeline.clamp_prompt_frames(round(float(self.duration.text()) * doc.FPS))
        except ValueError:
            frames = clip["end_frame"] - clip["start_frame"]
        self.resize_clip(clip, frames)
        self.duration.setText(f"{frames / doc.FPS:g}")

    def resize_clip(self, clip, frames):
        if frames != clip["end_frame"] - clip["start_frame"] and self.run is not None:
            self.run(timeline_edit.resize_prompt, clip["id"], frames)

    def slid(self, frames):
        clip = self.clip()
        if clip is None:
            return
        self.duration.setText(f"{frames / doc.FPS:g}")
        if self.slider.isSliderDown():
            if self.preview is not None:
                self.preview(clip["id"], frames)
        else:
            self.resize_clip(clip, frames)

    def slide_ended(self):
        clip = self.clip()
        if self.preview is not None:
            self.preview(None, None)
        if clip is not None:
            self.resize_clip(clip, self.slider.value())
