"""pieces of the panel ui."""

import time

import hou
from PySide6 import QtCore, QtGui, QtWidgets

from . import doc, motion_node, rig, timeline, timeline_edit

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
        p.setPen(self.palette().color(QtGui.QPalette.Text))
        for i, name in enumerate(TRACKS):
            p.drawText(
                QtCore.QRect(6, self.m.ruler_h + i * self.m.row_h, self.m.label_w - 8, self.m.row_h),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                name,
            )
        p.end()


class TimelineCanvas(FontSized):
    frame_requested = QtCore.Signal(int)
    edit_requested = QtCore.Signal()
    copy_requested = QtCore.Signal(int)
    paste_requested = QtCore.Signal(str, bool)

    def __init__(self):
        self.prompts, self.constraints, self.total, self.frame = [], [], 5 * doc.FPS, 0
        self.changed = set()
        super().__init__()

    def scale(self):
        return max(self.width() / max(self.total, 1), self.m.px_per_frame)

    def frame_x(self, frame):
        """middle of the frame's cell"""
        return (frame + 0.5) * self.scale()

    def apply_metrics(self):
        super().apply_metrics()
        self.setFixedHeight(self.m.tracks_h)
        self.setMinimumWidth(self.total * self.m.px_per_frame)

    def set_state(self, prompts, frame, constraints=()):
        total = prompts[-1]["end_frame"] if prompts else 5 * doc.FPS
        constraints = list(constraints)
        if (prompts, frame, total, constraints) != (self.prompts, self.frame, self.total, self.constraints):
            self.prompts, self.frame, self.total, self.constraints = prompts, frame, total, constraints
            self.setMinimumWidth(total * self.m.px_per_frame)
            self.update()


class TimelineArea(QtWidgets.QWidget):
    def __init__(self, canvas=None):
        super().__init__()
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.canvas = canvas or TimelineCanvas()
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
        self.text.setToolTip("")
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
        self.slider.valueChanged.connect(self.slid)
        self.slider.sliderReleased.connect(self.slide_ended)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

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


class KimodoPanel(QtWidgets.QWidget):
    def __init__(self, parm_editor_factory=None, canvas=None):
        super().__init__()
        self.node = None
        self.refreshing = False
        self.hover_hint = ""
        self.doc_text, self.doc = None, None
        self.parm_editor = None
        self.parm_editor_factory = parm_editor_factory or (lambda node: hou.qt.ParmDialog(node, compact=True))
        self.timeline = TimelineArea(canvas)
        canvas = self.timeline.canvas
        canvas.frame_requested.connect(lambda frame: hou.setFrame(frame + 1))
        canvas.edit_requested.connect(self.edit_pose)
        canvas.copy_requested.connect(lambda frame: self.with_node(motion_node.copy_pose, frame))
        canvas.paste_requested.connect(
            lambda ctype, keep_root: self.with_node(
                motion_node.paste_pose, ctype, int(round(hou.frame())) - 1, keep_root
            )
        )
        self.status = StatusWidget()
        self.status.dismissed.connect(self.dismiss_error)
        self.buttons = QtWidgets.QWidget()
        self.button_rows = QtWidgets.QVBoxLayout(self.buttons)
        self.button_rows.setSpacing(4)
        self.generate_button = QtWidgets.QPushButton("Generate")
        self.generate_button.clicked.connect(self.press_generate)
        self.button_rows.addWidget(self.generate_button, 0, QtCore.Qt.AlignLeft)
        self.buttons.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.clip_fields = ClipFields()
        self.divider = QtWidgets.QFrame()
        self.divider.setFrameShape(QtWidgets.QFrame.HLine)
        self.divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.divider_box = QtWidgets.QWidget()
        around = QtWidgets.QVBoxLayout(self.divider_box)
        around.setContentsMargins(0, DIVIDER_SPACE, 0, DIVIDER_SPACE)
        around.addWidget(self.divider)
        self.divider_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.clip_fields.run = self.edit_clip
        self.clip_fields.preview = self.preview_clip
        fixed = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.timeline.setSizePolicy(fixed)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        self.stack = QtWidgets.QVBoxLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.stack.setSpacing(4)
        self.stack.addWidget(self.timeline, 0)
        self.stack.addWidget(self.status, 0)
        self.stack.addWidget(self.clip_fields, 0)
        self.stack.addWidget(self.buttons, 0)
        self.stack.addWidget(self.divider_box, 0)
        self.stack.addStretch(1)
        self.place_buttons()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def tick(self):
        """refresh between cooks, never inside one"""
        motion_node.when_idle(("panel", id(self)), self.refresh_if_alive)

    def refresh_if_alive(self):
        try:
            self.refresh()
        except RuntimeError:
            pass

    def set_node(self, node):
        self.node = node
        if node is not None:
            if self.parm_editor is None:
                self.parm_editor = self.parm_editor_factory(node)
                self.stack.takeAt(self.stack.count() - 1)
                self.stack.addWidget(self.parm_editor, 1)
            else:
                self.parm_editor.setNode(node)
            motion_node.watch(node)
        self.refresh()

    def refresh(self):
        """cheap, runs five times a second"""
        if self.refreshing or not self.isVisible():
            return
        self.refreshing = True
        try:
            if self.node is None:
                return
            self.status.show_status(motion_node.get_status(self.node))
            self.generate_button.setText("Cancel" if self.node.evalParm("busy") else "Generate")
            self.clip_fields.refresh(self.node)
            self.place_buttons()
            self.status.show_hint(self.hover_hint or motion_node.pose_hint(self.node))
            text = self.node.parm("kimodo_doc").unexpandedString()
            if text != self.doc_text:
                self.doc_text, self.doc = text, motion_node.read_doc(self.node)
            d = self.doc
            canvas = self.timeline.canvas
            changed = motion_node.changed_keyframes(self.node)
            if changed != canvas.changed:
                canvas.changed = changed
                canvas.update()
            canvas.set_state(d["prompts"], int(round(hou.frame())) - 1, d["constraints"])
        except (hou.ObjectWasDeleted, doc.DocError):
            self.node = None
        finally:
            self.refreshing = False

    def buttons_right(self):
        """where the parm editor's menus end"""
        editor = self.parm_editor.mapTo(self, QtCore.QPoint(0, 0)).x() if self.parm_editor is not None else 0
        return editor + round(MENU_RIGHT * ui_scale())

    def place_buttons(self):
        left = self.timeline.labels.width()
        if self.button_rows.contentsMargins().left() != left:
            self.button_rows.setContentsMargins(left, 0, 0, 0)
        size = QtCore.QSize(
            max(self.buttons_right() - left, 50), round(QtWidgets.QPushButton().sizeHint().height() * GENERATE_TALLER)
        )
        if self.generate_button.size() != size:
            self.generate_button.setFixedSize(size)
        self.clip_fields.place(left)

    def edit_clip(self, fn, *args):
        self.with_node(fn, *args)
        self.refresh()

    def preview_clip(self, pid, frames):
        """while duration is dragged the timeline shows the clip that long"""
        canvas = self.timeline.canvas
        if not hasattr(canvas, "preview"):
            return
        if pid is None:
            canvas.preview = None
        else:
            shown = {"prompts": [dict(p) for p in canvas.prompts], "constraints": []}
            timeline.resize_prompt(shown, pid, frames)
            canvas.preview = shown["prompts"]
        canvas.update()

    def press_generate(self):
        """generate, or cancel while a generation runs"""
        if self.node is not None:
            self.with_node(motion_node.cancel if self.node.evalParm("busy") else motion_node.generate)
            self.refresh()

    def with_node(self, fn, *args):
        if self.node is not None:
            try:
                fn(self.node, *args)
            except hou.Error as e:
                motion_node.set_status(self.node, e.instanceMessage())
            except rig.NotReady as e:
                motion_node.set_status(self.node, f"The rig is not ready: {e}. Try again.")
            self.refresh()

    def edit_pose(self):
        if self.node is not None:
            motion_node.edit_pose(self.node)
            self.refresh()

    def dismiss_error(self):
        if self.node is not None:
            motion_node.clear_error(self.node)
            self.refresh()

    def close_panel(self):
        self.timer.stop()
        self.node = None
