"""the timeline widget and the panel around it."""

import copy

import hou
from PySide6 import QtCore, QtGui, QtWidgets

from . import doc, motion_node, panel_parts, rig, timeline, timeline_edit

TAIL = 2 * doc.FPS
EDGE = 5

HINTS = {
    "prompts": "Prompts: click empty space to add · double-click to edit · drag the right edge for length · "
    "right-click to delete",
    "constraints": "Constraints: click a track for a keyframe at the current frame · Ctrl+drag for a range · "
    "drag to move · right-click to delete or change type",
    "ruler": "Ruler: click or drag to set the frame · wheel pans · Shift+wheel zooms · Space plays · "
    "Left/Right step a frame",
    "": "",
}
ZOOM_STEP, ZOOM_MIN, ZOOM_MAX = 1.25, 0.3, 8.0


class Canvas(panel_parts.TimelineCanvas):
    prompt_add_requested = QtCore.Signal()
    prompt_text_requested = QtCore.Signal(str, str)
    prompt_resize_requested = QtCore.Signal(str, int)
    prompt_delete_requested = QtCore.Signal(str)
    run_move_requested = QtCore.Signal(str, object, int)
    run_delete_requested = QtCore.Signal(str, object)
    range_requested = QtCore.Signal(str, int, int)
    heading_requested = QtCore.Signal(str, bool)
    smooth_path_requested = QtCore.Signal(bool)
    play_requested = QtCore.Signal()
    step_requested = QtCore.Signal(int)
    hint_changed = QtCore.Signal(str)

    def __init__(self):
        self.drag = None
        self.preview = None
        self.editor, self.editing = None, None
        self.hint = ""
        self.zoom = 1.0
        self.smooth_path = False
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def span(self):
        return self.total + TAIL

    def scale(self):
        return max(self.width() / self.span(), self.m.px_per_frame)

    def apply_metrics(self):
        super().apply_metrics()
        self.m.px_per_frame = max(1, round(self.m.px_per_frame * self.zoom))
        self.setMinimumWidth(self.span() * self.m.px_per_frame)

    def set_state(self, prompts, frame, constraints=()):
        before = self.frame
        super().set_state(prompts, frame, constraints)
        self.setMinimumWidth(self.span() * self.m.px_per_frame)
        if self.frame != before:
            self.follow()

    def set_smooth_path(self, on):
        if on != self.smooth_path:
            self.smooth_path = on
            self.update()

    def set_frame(self, frame):
        if frame != self.frame:
            self.frame = frame
            self.update()
            self.follow()

    def scroll_area(self):
        area = self.parentWidget().parentWidget() if self.parentWidget() is not None else None
        return area if isinstance(area, QtWidgets.QScrollArea) else None

    def follow(self):
        """scroll so the playhead shows, unless something is dragged"""
        area = self.scroll_area()
        if area is None or self.drag is not None:
            return
        bar, width, x = area.horizontalScrollBar(), area.viewport().width(), self.frame_x(self.frame)
        if not bar.value() + 8 <= x <= bar.value() + width - 8:
            bar.setValue(int(x - width / 2))

    def set_zoom(self, zoom, anchor_x):
        area = self.scroll_area()
        frame = anchor_x / self.scale()
        offset = anchor_x - (area.horizontalScrollBar().value() if area else 0)
        self.zoom = min(max(zoom, ZOOM_MIN), ZOOM_MAX)
        self.apply_metrics()
        if area is not None:
            self.resize(max(area.viewport().width(), self.minimumWidth()), self.height())
            area.horizontalScrollBar().setValue(int(frame * self.scale() - offset))
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        area = self.scroll_area()
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            self.set_zoom(self.zoom * ZOOM_STEP ** (delta / 120.0), event.position().x())
        elif area is not None:
            bar = area.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.play_requested.emit()
        elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            self.step_requested.emit(-1 if event.key() == QtCore.Qt.Key_Left else 1)
        else:
            super().keyPressEvent(event)

    def row_at(self, pos):
        """-1 ruler, 0 clips, 1.. tracks, None below"""
        row = (pos.y() - self.m.ruler_h) // self.m.row_h if pos.y() >= self.m.ruler_h else -1
        return row if row < len(panel_parts.TRACKS) else None

    def shown_prompts(self):
        return self.preview if self.preview is not None else self.prompts

    def prompt_rect(self, pr):
        scale = self.scale()
        x0, x1 = int(pr["start_frame"] * scale), int(pr["end_frame"] * scale)
        return QtCore.QRect(x0 + 1, self.m.ruler_h + 2, x1 - x0 - 2, self.m.row_h - 5)

    def prompt_at(self, pos):
        scale = self.scale()
        for pr in self.shown_prompts():
            if abs(pos.x() - pr["end_frame"] * scale) <= EDGE:
                return pr, True
        for pr in self.shown_prompts():
            if pr["start_frame"] * scale <= pos.x() < pr["end_frame"] * scale:
                return pr, False
        return None, False

    def frame_at(self, x):
        return min(max(int(x // self.scale()), 0), self.total - 1)

    def run_at(self, pos):
        """(constraint, run) under a point, keyframes win over ranges"""
        row = self.row_at(pos)
        if not row or row < 1:
            return None, None
        ctype, scale = panel_parts.TRACK_TYPES[row - 1], self.scale()
        reach = max(self.m.row_h // 2, 6)
        found = None
        for c in self.constraints:
            if c["type"] != ctype:
                continue
            for run in timeline.runs(c["frames"]):
                if len(run) == 1 and abs(pos.x() - self.frame_x(run[0])) <= reach:
                    return c, run
                if len(run) > 1 and run[0] * scale - 2 <= pos.x() <= (run[-1] + 1) * scale + 2:
                    found = c, run
        return found or (None, None)

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        self.setFocus()
        row = self.row_at(pos)
        if row and row >= 1:
            return self.press_track(event, pos, panel_parts.TRACK_TYPES[row - 1])
        if row == -1 and event.button() == QtCore.Qt.LeftButton:
            self.drag = {"kind": "scrub"}
            self.frame_requested.emit(self.frame_at(pos.x()))
        if row != 0:
            return
        pr, edge = self.prompt_at(pos)
        if event.button() == QtCore.Qt.RightButton and pr is not None:
            menu = QtWidgets.QMenu(self)
            edit = menu.addAction("Edit Text")
            delete = menu.addAction("Delete")
            delete.setEnabled(len(self.prompts) > 1)
            chosen = menu.exec(event.globalPosition().toPoint())
            if chosen is edit:
                self.edit_text(pr["id"])
            elif chosen is delete:
                self.prompt_delete_requested.emit(pr["id"])
        elif event.button() == QtCore.Qt.LeftButton:
            if edge:
                self.drag = {
                    "kind": "resize",
                    "id": pr["id"],
                    "start": pr["start_frame"],
                    "frames": pr["end_frame"] - pr["start_frame"],
                }
            elif pr is None:
                self.drag = {"kind": "add"}

    def press_track(self, event, pos, ctype):
        c, run = self.run_at(pos)
        if event.button() == QtCore.Qt.RightButton and (c is not None or ctype == "root2d"):
            menu = QtWidgets.QMenu(self)
            delete = heading = smooth = None
            types = {}
            if c is not None:
                delete = menu.addAction("Delete")
                menu.addSeparator()
                types = {
                    menu.addAction(f"Change to {panel_parts.TRACKS[i + 1]}"): t
                    for i, t in enumerate(panel_parts.TRACK_TYPES)
                    if t != ctype
                }
                menu.addSeparator()
            if c is not None and ctype == "root2d":
                heading = menu.addAction("Constrain Heading")
                heading.setCheckable(True)
                heading.setChecked(bool(c.get("heading")))
            if ctype == "root2d":
                smooth = menu.addAction("Smooth Path Through Keyframes")
                smooth.setCheckable(True)
                smooth.setChecked(self.smooth_path)
            chosen = menu.exec(event.globalPosition().toPoint())
            if chosen is None:
                pass
            elif chosen is delete:
                self.run_delete_requested.emit(c["id"], run)
            elif chosen is heading:
                self.heading_requested.emit(c["id"], not c.get("heading"))
            elif chosen is smooth:
                self.smooth_path_requested.emit(not self.smooth_path)
            elif chosen in types:
                self.type_requested.emit(c["id"], types[chosen])
        elif event.button() == QtCore.Qt.LeftButton:
            frame = self.frame_at(pos.x())
            if event.modifiers() & QtCore.Qt.ControlModifier:
                start = run[0] if run is not None and len(run) == 1 else frame
                self.drag = {"kind": "range", "type": ctype, "first": start, "last": frame}
            elif c is not None:
                self.drag = {"kind": "move", "c": c, "run": run, "x": pos.x(), "delta": 0, "moved": False, "ok": True}
            else:
                self.drag = {"kind": "key", "type": ctype}
            self.update()

    def drag_track(self, drag, pos):
        if drag["kind"] == "range":
            drag["last"] = self.frame_at(pos.x())
            lo, hi = sorted((drag["first"], drag["last"]))
            self.set_hint(f"Range: frames {lo + 1} to {hi + 1}")
        else:
            drag["moved"] = drag["moved"] or abs(pos.x() - drag["x"]) > 3
            if not drag["moved"]:
                return
            run = drag["run"]
            delta = round((pos.x() - drag["x"]) / self.scale())
            delta = min(max(delta, -run[0]), self.total - 1 - run[-1])
            shown = {"prompts": self.prompts, "constraints": copy.deepcopy(self.constraints)}
            drag["delta"] = delta
            drag["ok"] = delta == 0 or bool(timeline.move_run(shown, drag["c"]["id"], run, delta))
            self.set_hint(
                f"Move to frame {run[0] + delta + 1}" + ("" if drag["ok"] else ": a constraint is in the way")
            )
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        drag = self.drag
        if drag is not None and drag["kind"] in ("range", "move"):
            self.drag_track(drag, pos)
        elif drag is not None and drag["kind"] == "scrub":
            self.frame_requested.emit(self.frame_at(pos.x()))
        elif drag is None:
            row = self.row_at(pos)
            edge = row == 0 and self.prompt_at(pos)[1]
            self.setCursor(QtCore.Qt.SizeHorCursor if edge else QtCore.Qt.ArrowCursor)
            self.set_hint("prompts" if row == 0 else "ruler" if row == -1 else "constraints" if row else "")
        elif drag["kind"] == "resize":
            frames = timeline.clamp_prompt_frames(round(pos.x() / self.scale()) - drag["start"])
            shown = {"prompts": copy.deepcopy(self.prompts), "constraints": []}
            timeline.resize_prompt(shown, drag["id"], frames)
            drag["frames"], self.preview = frames, shown["prompts"]
            self.set_hint(f"{frames / doc.FPS:.1f} s")
            self.update()

    def mouseReleaseEvent(self, event):
        drag, self.drag, self.preview = self.drag, None, None
        if drag is None:
            return
        pos = event.position().toPoint()
        if drag["kind"] == "resize":
            before = next(p["end_frame"] - p["start_frame"] for p in self.prompts if p["id"] == drag["id"])
            if drag["frames"] != before:
                self.prompt_resize_requested.emit(drag["id"], drag["frames"])
        elif drag["kind"] == "add" and self.row_at(pos) == 0 and self.prompt_at(pos)[0] is None:
            self.prompt_add_requested.emit()
        elif drag["kind"] == "key" or (drag["kind"] == "range" and drag["first"] == drag["last"]):
            self.add_requested.emit(drag["type"])
        elif drag["kind"] == "range":
            self.range_requested.emit(drag["type"], drag["first"], drag["last"])
        elif drag["kind"] == "move" and not drag["moved"]:
            x = pos.x() / self.scale() - 0.5
            self.frame_requested.emit(min(drag["run"], key=lambda f: abs(f - x)))
        elif drag["kind"] == "move" and drag["ok"] and drag["delta"]:
            self.run_move_requested.emit(drag["c"]["id"], drag["run"], drag["delta"])
        self.update()
        self.mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        pos = event.position().toPoint()
        pr, edge = self.prompt_at(pos) if self.row_at(pos) == 0 else (None, False)
        if pr is not None and not edge:
            self.edit_text(pr["id"])

    def leaveEvent(self, event):
        self.set_hint("")

    def set_hint(self, key):
        text = HINTS.get(key, key)
        if text != self.hint:
            self.hint = text
            self.hint_changed.emit(text)

    def edit_text(self, pid):
        self.close_editor()
        pr = next((p for p in self.prompts if p["id"] == pid), None)
        if pr is None:
            return
        editor = QtWidgets.QLineEdit(pr["text"], self)
        editor.setGeometry(self.prompt_rect(pr).adjusted(0, -2, 0, 2))
        editor.selectAll()
        editor.installEventFilter(self)
        editor.editingFinished.connect(lambda: self.finish_edit(pid))
        editor.show()
        editor.setFocus()
        self.editor, self.editing = editor, pid

    def finish_edit(self, pid):
        editor = self.editor
        if editor is None:
            return
        text = editor.text().strip()
        self.close_editor()
        old = next((p["text"] for p in self.prompts if p["id"] == pid), None)
        if text and old is not None and text != old:
            self.prompt_text_requested.emit(pid, text)

    def close_editor(self):
        editor, self.editor = self.editor, None
        if editor is not None:
            editor.hide()
            editor.deleteLater()

    def eventFilter(self, obj, event):
        if obj is self.editor and event.type() in (QtCore.QEvent.KeyPress, QtCore.QEvent.ShortcutOverride):
            key = event.key()
            if key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if event.type() == QtCore.QEvent.KeyPress:
                    self.finish_edit(self.editing) if key != QtCore.Qt.Key_Escape else self.close_editor()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        pal = self.palette()
        ruler_h, row_h, scale = self.m.ruler_h, self.m.row_h, self.scale()
        text = pal.color(QtGui.QPalette.Text)
        prompts = self.shown_prompts()
        end_x = int((prompts[-1]["end_frame"] if prompts else self.total) * scale)
        for i in range(len(panel_parts.TRACKS)):
            role = QtGui.QPalette.AlternateBase if i % 2 else QtGui.QPalette.Base
            p.fillRect(0, ruler_h + i * row_h, self.width(), row_h - 1, pal.color(role))
        p.fillRect(end_x, 0, self.width() - end_x, self.height(), QtGui.QColor(0, 0, 0, 70))
        self.paint_ruler(p, text, end_x)
        for pr in prompts:
            self.paint_prompt(p, pr)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        waypoints = sorted(f for c in self.constraints if c["type"] == "root2d" for f in c["frames"])
        if self.smooth_path and len(waypoints) > 1:
            self.paint_constraint(p, {"type": "root2d"}, list(range(waypoints[0], waypoints[-1] + 1)), 70)
        drag = self.drag or {"kind": ""}
        moving = drag["kind"] == "move" and drag["moved"]
        for c in self.constraints:
            if moving and c["id"] == drag["c"]["id"]:
                self.paint_constraint(p, c, [f for f in c["frames"] if f not in drag["run"]])
                self.paint_constraint(p, c, [f + drag["delta"] for f in drag["run"]], 255 if drag["ok"] else 90)
            else:
                self.paint_constraint(p, c)
        if drag["kind"] == "range":
            lo, hi = sorted((drag["first"], drag["last"]))
            self.paint_constraint(p, {"type": drag["type"]}, list(range(lo, hi + 1)), 150)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        x = int(self.frame_x(self.frame))
        p.setPen(QtGui.QPen(QtGui.QColor(230, 90, 60), 2))
        p.drawLine(x, 0, x, self.height())
        p.end()

    def paint_ruler(self, p, color, end_x):
        ruler_h, scale = self.m.ruler_h, self.scale()
        label_w = p.fontMetrics().horizontalAdvance("00s") + 8
        step = next(s for s in (1, 2, 5, 10, 30, 60) if s * doc.FPS * scale >= label_w or s == 60)
        p.setPen(color)
        for second in range(self.span() // doc.FPS + 1):
            x = int(self.frame_x(second * doc.FPS))
            if x > end_x:
                break
            p.drawLine(x, ruler_h - (6 if second % step == 0 else 3), x, ruler_h - 1)
            if second % step == 0:
                p.drawText(x + 3, ruler_h - 6, f"{second}s")

    def paint_prompt(self, p, pr):
        rect = self.prompt_rect(pr)
        p.fillRect(rect, QtGui.QColor(70, 120, 200))
        p.fillRect(rect.right() - 2, rect.top(), 3, rect.height(), QtGui.QColor(150, 190, 245))
        p.setPen(QtGui.QColor(255, 255, 255))
        fm = p.fontMetrics()
        length = f"{(pr['end_frame'] - pr['start_frame']) / doc.FPS:.1f} s"
        inner = rect.adjusted(4, 0, -6, 0)
        if inner.width() > fm.horizontalAdvance(length) * 3:
            p.drawText(inner, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, length)
            inner.setRight(inner.right() - fm.horizontalAdvance(length) - 8)
        p.drawText(
            inner,
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            fm.elidedText(pr["text"], QtCore.Qt.ElideRight, inner.width()),
        )

    def track_y(self, ctype):
        return self.m.ruler_h + (panel_parts.TRACK_TYPES.index(ctype) + 1) * self.m.row_h + self.m.row_h // 2

    def paint_constraint(self, p, c, frames=None, alpha=255):
        scale, y = self.scale(), self.track_y(c["type"])
        size = max(self.m.row_h // 2 - 2, 4)
        color = QtGui.QColor(240, 210, 130, alpha) if c.get("auto") else QtGui.QColor(240, 170, 60, alpha)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(color)
        for run in timeline.runs(c["frames"] if frames is None else frames):
            if len(run) > 1:
                p.drawRoundedRect(QtCore.QRectF(run[0] * scale, y - size / 2, len(run) * scale, size), 2, 2)
                mark = QtCore.QLineF(run[0] * scale + 2, y, (run[-1] + 1) * scale - 2, y)
            else:
                cx = self.frame_x(run[0])
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
                mark = QtCore.QLineF(cx - size + 2, y, cx + size - 2, y)
            if c.get("heading"):
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, alpha), 1.5))
                p.drawLine(mark)
                p.setPen(QtCore.Qt.NoPen)


class Panel(panel_parts.KimodoPanel):
    def __init__(self, parm_editor_factory=None):
        super().__init__(parm_editor_factory, canvas=Canvas())
        canvas = self.timeline.canvas
        canvas.prompt_add_requested.connect(self.add_prompt)
        canvas.prompt_text_requested.connect(lambda pid, text: self.edit(timeline_edit.set_prompt_text, pid, text))
        canvas.prompt_resize_requested.connect(lambda pid, n: self.edit(timeline_edit.resize_prompt, pid, n))
        canvas.prompt_delete_requested.connect(lambda pid: self.edit(timeline_edit.delete_prompt, pid))
        canvas.run_move_requested.connect(lambda cid, run, delta: self.edit(timeline_edit.move_run, cid, run, delta))
        canvas.run_delete_requested.connect(lambda cid, run: self.edit(timeline_edit.delete_run, cid, run))
        canvas.range_requested.connect(lambda ctype, a, b: self.edit(timeline_edit.add_range, ctype, a, b))
        canvas.heading_requested.connect(lambda cid, on: self.edit(timeline_edit.set_heading, cid, on))
        canvas.smooth_path_requested.connect(lambda on: self.edit(timeline_edit.set_smooth_path, on))
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        self.hint.setEnabled(False)
        self.hint.setMinimumHeight(self.hint.fontMetrics().height())
        canvas.hint_changed.connect(self.hint.setText)
        self.status.layout().insertWidget(0, self.hint)
        canvas.play_requested.connect(self.toggle_play)
        canvas.step_requested.connect(lambda step: hou.setFrame(max(int(round(hou.frame())) + step, 1)))
        self.generate_button = QtWidgets.QPushButton("Generate")
        self.generate_button.setToolTip("Generate with the node's settings; Cancel while it runs.")
        self.generate_button.clicked.connect(self.generate_or_cancel)
        self.lower_layout.itemAt(1).layout().insertWidget(0, self.generate_button)
        self.guide_key = None
        self.playbar_callback = None
        if hou.isUIAvailable():
            self.playbar_callback = self.on_playbar
            hou.playbar.addEventCallback(self.playbar_callback)

    def refresh(self):
        """cheap, runs five times a second"""
        super().refresh()
        node = self.node
        try:
            busy = node is not None and bool(node.evalParm("busy"))
        except hou.ObjectWasDeleted:
            node, busy = None, False
        self.timeline.setEnabled(node is not None)
        self.generate_button.setEnabled(node is not None)
        self.generate_button.setText("Cancel" if busy else "Generate")
        if node is None:
            self.hint.setText("Select a Kimodo Motion node.")
        else:
            d = motion_node.read_doc(node)
            self.timeline.canvas.set_smooth_path(bool(d.get("smooth_root_path")))
            key = (d["constraints"], d.get("smooth_root_path"), node.evalParm("unapplied"), node.evalParm("rig_npz"))
            if key != self.guide_key and not motion_node.is_editing(node):
                self.guide_key = key
                timeline_edit.update_root_path_guide(node)

    def on_playbar(self, event_type, frame):
        if event_type == hou.playbarEvent.FrameChanged:
            self.timeline.canvas.set_frame(int(round(frame)) - 1)

    def toggle_play(self):
        if hou.isUIAvailable():
            hou.playbar.stop() if hou.playbar.isPlaying() else hou.playbar.play()

    def generate_or_cancel(self):
        if self.node is not None:
            motion_node.cancel(self.node) if self.node.evalParm("busy") else motion_node.generate(self.node)
            self.refresh()

    def close_panel(self):
        if self.playbar_callback is not None:
            hou.playbar.removeEventCallback(self.playbar_callback)
            self.playbar_callback = None
        super().close_panel()

    def edit(self, fn, *args):
        """run a timeline edit, errors go into the node's status"""
        if self.node is None:
            return None
        result = None
        try:
            result = fn(self.node, *args)
        except hou.Error as e:
            motion_node.set_status(self.node, e.instanceMessage())
        except (rig.NotReady, doc.DocError) as e:
            motion_node.set_status(self.node, f"Not done: {e}.")
        self.refresh()
        return result

    def add_prompt(self):
        pid = self.edit(timeline_edit.add_prompt)
        if pid:
            self.timeline.canvas.edit_text(pid)
