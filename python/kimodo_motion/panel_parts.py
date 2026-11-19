"""pieces of the panel ui."""

import time

import hou
from PySide6 import QtCore, QtGui, QtWidgets

from . import doc, motion_node, rig

TRACKS = ("Prompts", "Full-Body", "2D Root", "Left Hand", "Right Hand", "Left Foot", "Right Foot")
TRACK_TYPES = ("fullbody", "root2d", "left-hand", "right-hand", "left-foot", "right-foot")
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


def paste_entries(ctype, frame):
    """[(text, keep the frame's root)] for a track's paste entries"""
    n = frame + 1
    if ctype == "fullbody":
        return [(f"Paste Pose and Root at Frame {n}", False), (f"Paste Pose Only at Frame {n}", True)]
    if ctype == "root2d":
        return [(f"Paste Root at Frame {n}", False)]
    return [(f"Paste Pose at Frame {n}", False)]


def houdini_tooltip(widget, title, text, hotkey=""):
    widget.setProperty("tooltip_text", f"{title}: {text}" + (f" Hotkey: {hotkey}." if hotkey else ""))
    if not hou.isUIAvailable():
        widget.setToolTip(text)
        return None
    tip = hou.qt.ToolTip()
    tip.setTitle(title)
    tip.setText(text)
    if hotkey:
        tip.setHotkey(hotkey)
    tip.setTargetWidget(widget)
    return tip


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
    add_requested = QtCore.Signal(str)
    delete_requested = QtCore.Signal(str)
    type_requested = QtCore.Signal(str, str)
    frame_requested = QtCore.Signal(int)
    edit_requested = QtCore.Signal()
    copy_requested = QtCore.Signal(int)
    paste_requested = QtCore.Signal(str, bool)

    def __init__(self):
        self.prompts, self.constraints, self.total, self.frame = [], [], 5 * doc.FPS, 0
        super().__init__()

    def scale(self):
        return max(self.width() / max(self.total, 1), self.m.px_per_frame)

    def frame_x(self, frame):
        """middle of the frame's cell"""
        return (frame + 0.5) * self.scale()

    def constraint_at(self, pos):
        row = (pos.y() - self.m.ruler_h) // self.m.row_h - 1
        if not 0 <= row < len(TRACK_TYPES):
            return None, None
        ctype = TRACK_TYPES[row]
        reach = max(self.m.row_h // 2, 6)
        for c in self.constraints:
            if c["type"] == ctype and any(abs(pos.x() - self.frame_x(f)) <= reach for f in c["frames"]):
                return c, ctype
        return None, ctype

    def track_menu(self, constraint, ctype, frame):
        menu, actions = QtWidgets.QMenu(self), {}
        if constraint is not None:
            actions[menu.addAction("Copy Pose")] = lambda: self.copy_requested.emit(frame)
        for text, keep_root in paste_entries(ctype, self.frame):
            action = menu.addAction(text)
            action.setEnabled(motion_node.has_copied_pose())
            actions[action] = lambda keep_root=keep_root: self.paste_requested.emit(ctype, keep_root)
        if constraint is not None:
            menu.addSeparator()
            actions[menu.addAction("Delete")] = lambda: self.delete_requested.emit(constraint["id"])
            menu.addSeparator()
            for i, t in enumerate(TRACK_TYPES):
                if t != constraint["type"]:
                    actions[menu.addAction(f"Change to {TRACKS[i + 1]}")] = lambda t=t: self.type_requested.emit(
                        constraint["id"], t
                    )
        return menu, actions

    def nearest_frame(self, constraint, event):
        x = event.position().x() / self.scale() - 0.5
        return min(constraint["frames"], key=lambda f: abs(f - x))

    def mousePressEvent(self, event):
        c, ctype = self.constraint_at(event.position().toPoint())
        if ctype is None:
            return super().mousePressEvent(event)
        if event.button() == QtCore.Qt.RightButton:
            menu, actions = self.track_menu(c, ctype, self.nearest_frame(c, event) if c is not None else None)
            chosen = menu.exec(event.globalPosition().toPoint())
            if chosen in actions:
                actions[chosen]()
        elif event.button() == QtCore.Qt.LeftButton:
            if c is not None:
                self.frame_requested.emit(self.nearest_frame(c, event))
            else:
                self.add_requested.emit(ctype)

    def mouseDoubleClickEvent(self, event):
        c, _ = self.constraint_at(event.position().toPoint())
        if c is not None and event.button() == QtCore.Qt.LeftButton:
            self.frame_requested.emit(self.nearest_frame(c, event))
            self.edit_requested.emit()
        else:
            super().mouseDoubleClickEvent(event)

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

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        pal = self.palette()
        ruler_h, row_h = self.m.ruler_h, self.m.row_h
        scale = self.scale()
        text = pal.color(QtGui.QPalette.Text)
        for i in range(len(TRACKS)):
            role = QtGui.QPalette.AlternateBase if i % 2 else QtGui.QPalette.Base
            p.fillRect(0, ruler_h + i * row_h, self.width(), row_h - 1, pal.color(role))
        p.setPen(text)
        for second in range(self.total // doc.FPS + 1):
            x = int(self.frame_x(second * doc.FPS))
            p.drawLine(x, ruler_h - 6, x, ruler_h - 1)
            p.drawText(x + 3, ruler_h - 6, f"{second}s")
        for pr in self.prompts:
            x0, x1 = int(pr["start_frame"] * scale), int(pr["end_frame"] * scale)
            rect = QtCore.QRect(x0 + 1, ruler_h + 2, x1 - x0 - 2, row_h - 5)
            p.fillRect(rect, QtGui.QColor(70, 120, 200))
            p.setPen(QtGui.QColor(255, 255, 255))
            label = p.fontMetrics().elidedText(pr["text"], QtCore.Qt.ElideRight, rect.width() - 8)
            p.drawText(rect.adjusted(4, 0, -4, 0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        size = max(row_h // 2 - 2, 4)
        for c in self.constraints:
            y = ruler_h + (TRACK_TYPES.index(c["type"]) + 1) * row_h + row_h // 2
            frames = c["frames"]
            color = QtGui.QColor(*(motion_node.AUTO_MARKER_RGB if c.get("auto") else motion_node.MARKER_RGB))
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(color)
            if len(frames) > 1 and frames[-1] - frames[0] == len(frames) - 1:
                p.drawRoundedRect(QtCore.QRectF(frames[0] * scale, y - size / 2, (len(frames)) * scale, size), 2, 2)
                continue
            for f in frames:
                cx = self.frame_x(f)
                p.drawPolygon(
                    QtGui.QPolygonF(
                        [
                            QtCore.QPointF(cx, y - size),
                            QtCore.QPointF(cx + size, y),
                            QtCore.QPointF(cx, y + size),
                            QtCore.QPointF(cx - size, y),
                        ]
                    )
                )
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        x = int(self.frame_x(self.frame))
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
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {STAGE_COLORS['idle']};")
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
        self.tips = [houdini_tooltip(dismiss, "Dismiss", "Clear the warning or error on the node.")]
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


class KimodoPanel(QtWidgets.QWidget):
    def __init__(self, parm_editor_factory=None):
        super().__init__()
        self.node = None
        self.refreshing = False
        self.doc_text, self.doc = None, None
        self.parm_editor = None
        self.parm_editor_factory = parm_editor_factory or (lambda node: hou.qt.ParmDialog(node, compact=True))
        self.timeline = TimelineArea()
        canvas = self.timeline.canvas
        canvas.add_requested.connect(
            lambda ctype: self.with_node(motion_node.add_constraint, ctype, int(round(hou.frame())) - 1)
        )
        canvas.delete_requested.connect(lambda cid: self.with_node(motion_node.delete_constraint, cid))
        canvas.type_requested.connect(lambda cid, ctype: self.with_node(motion_node.set_constraint_type, cid, ctype))
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
                self.lower_layout.takeAt(2)
                self.lower_layout.addWidget(self.parm_editor, 1)
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
            self.status.show_hint(motion_node.pose_hint(self.node))
            text = self.node.parm("kimodo_doc").unexpandedString()
            if text != self.doc_text:
                self.doc_text, self.doc = text, motion_node.read_doc(self.node)
            d = self.doc
            self.timeline.canvas.set_state(d["prompts"], int(round(hou.frame())) - 1, d["constraints"])
        except (hou.ObjectWasDeleted, doc.DocError):
            self.node = None
        finally:
            self.refreshing = False

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
