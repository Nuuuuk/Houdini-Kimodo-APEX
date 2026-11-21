"""the timeline widget and the panel around it."""

import copy

import hou
from PySide6 import QtCore, QtGui, QtWidgets

from . import doc, motion_node, panel_parts, rig, timeline, timeline_edit

TAIL = 2 * doc.FPS
EDGE = 5
PLAYHEAD_REACH = 4

HINTS = {
    "prompts": "Prompts: click empty space to add · double-click to edit · drag the right edge for length · "
    "right-click to delete",
    "constraints": "Constraints: click a track for a keyframe at that frame · click a marker to select it · "
    "Shift+click to add or remove · drag empty space to select · drag to move · "
    "Ctrl+drag for a range · right-click to reset, delete or change type",
    "ruler": "Ruler: click or drag to set the frame · wheel pans · Shift+wheel zooms · Space plays · "
    "Left/Right step a frame",
    "": "",
}
ZOOM_STEP, ZOOM_MIN, ZOOM_MAX = 1.25, 0.3, 8.0
CLIP_RGB = (70, 120, 200)
OUTLINE = 2.0
MOVES_KEPT = 50


def outline_width():
    return OUTLINE * (hou.ui.globalScaleFactor() if hou.isUIAvailable() else 1.0)


def constraints_key(constraints):
    return tuple(sorted((c["id"], tuple(c["frames"])) for c in constraints))


class Canvas(panel_parts.TimelineCanvas):
    prompt_add_requested = QtCore.Signal()
    prompt_text_requested = QtCore.Signal(str, str)
    prompt_resize_requested = QtCore.Signal(str, int)
    prompt_delete_requested = QtCore.Signal(str)
    prompt_select_requested = QtCore.Signal(str)
    add_at_requested = QtCore.Signal(str, int)
    runs_move_requested = QtCore.Signal(object, int)
    selection_ended = QtCore.Signal()
    reset_requested = QtCore.Signal(object)
    run_delete_requested = QtCore.Signal(str, object)
    range_requested = QtCore.Signal(str, int, int)
    heading_requested = QtCore.Signal(str, bool)
    play_requested = QtCore.Signal()
    step_requested = QtCore.Signal(int)
    hint_changed = QtCore.Signal(str)

    def __init__(self):
        self.drag = None
        self.preview = None
        self.editor, self.editing = None, None
        self.selected = None
        self.marked = set()
        self.shifting = False
        self.resettable = False
        self.moves = []
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
        there = {(c["id"], f) for c in self.constraints for f in c["frames"]}
        if not self.marked <= there:
            self.marked = self.follow_moves() & there
            self.update()
        self.setMinimumWidth(self.span() * self.m.px_per_frame)
        if self.frame != before:
            self.follow()

    def follow_moves(self):
        """after an undo or redo of a move the selection stays on the markers"""
        key = constraints_key(self.constraints)
        for before, after, marked_before, marked_after in reversed(self.moves):
            if key == before and self.marked == marked_after:
                return marked_before
            if key == after and self.marked == marked_before:
                return marked_after
        return self.marked

    def remember_move(self, before, marked_before):
        self.moves = (self.moves + [(before, constraints_key(self.constraints), marked_before, set(self.marked))])[
            -MOVES_KEPT:
        ]

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

    def busy_selecting(self):
        """a box or move is dragged, or shift is held since a shift+click"""
        if (
            self.shifting
            and hou.isUIAvailable()
            and not QtWidgets.QApplication.queryKeyboardModifiers() & QtCore.Qt.ShiftModifier
        ):
            self.shifting = False
        return self.drag is not None or self.shifting

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Shift:
            self.shifting = False
            self.selection_ended.emit()
        super().keyReleaseEvent(event)

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

    def is_marked(self, c, run):
        return any((c["id"], f) in self.marked for f in run)

    def can_reset(self, runs):
        return self.resettable and any(self.is_changed(c, run) for c, run in runs)

    def is_changed(self, c, run):
        """differs from what the shown result was made with"""
        return any((c["type"], f) in self.changed for f in run)

    def marked_runs(self):
        return [(c, run) for c in self.constraints for run in timeline.runs(c["frames"]) if self.is_marked(c, run)]

    def marker_rect(self, ctype, run):
        scale, y, size = self.scale(), self.track_y(ctype), max(self.m.row_h // 2 - 2, 4)
        if len(run) > 1:
            return QtCore.QRectF(run[0] * scale, y - size / 2, len(run) * scale, size)
        return QtCore.QRectF(self.frame_x(run[0]) - size, y - size, 2 * size, 2 * size)

    def runs_in(self, rect):
        return {
            (c["id"], f)
            for c in self.constraints
            for run in timeline.runs(c["frames"])
            if self.marker_rect(c["type"], run).intersects(rect)
            for f in run
        }

    def moved_marks(self, runs, delta):
        moving = {(cid, f) for cid, run in runs for f in run}
        return {(cid, f + delta) if (cid, f) in moving else (cid, f) for cid, f in self.marked}

    def on_playhead(self, pos):
        """on the current frame's line with nothing else under it"""
        row = self.row_at(pos)
        if abs(pos.x() - self.frame_x(self.frame)) > PLAYHEAD_REACH or row is None or row == -1:
            return False
        return self.prompt_at(pos)[0] is None if row == 0 else self.run_at(pos)[0] is None

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        self.setFocus()
        row = self.row_at(pos)
        self.shifting = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        if event.button() == QtCore.Qt.LeftButton and not event.modifiers() and self.on_playhead(pos):
            self.drag = {
                "kind": "playhead",
                "type": panel_parts.TRACK_TYPES[row - 1] if row >= 1 else None,
                "frame": self.frame,
                "x": pos.x(),
                "moved": False,
            }
            return
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
            elif pr["id"] != self.selected:
                self.prompt_select_requested.emit(pr["id"])

    def press_track(self, event, pos, ctype):
        c, run = self.run_at(pos)
        if event.button() == QtCore.Qt.RightButton:
            x = pos.x() / self.scale() - 0.5
            menu, actions = self.track_menu(c, ctype, min(run, key=lambda f: abs(f - x)) if c is not None else None)
            for i, action in enumerate(menu.actions()):
                if action.text() == "Delete":
                    actions[action] = lambda: self.run_delete_requested.emit(c["id"], run)
                    reset = QtGui.QAction("Reset Constraint", menu)
                    reset.setEnabled(self.can_reset([(c, run)]))
                    menu.insertAction(menu.actions()[i + 1] if i + 1 < len(menu.actions()) else None, reset)
                    actions[reset] = lambda: self.reset_requested.emit([(c["id"], run)])
                    break
            if ctype == "root2d" and c is not None:
                menu.addSeparator()
                heading = menu.addAction("Constrain Heading")
                heading.setCheckable(True)
                heading.setChecked(bool(c.get("heading")))
                actions[heading] = lambda: self.heading_requested.emit(c["id"], not c.get("heading"))
            chosen = menu.exec(event.globalPosition().toPoint())
            if chosen in actions:
                actions[chosen]()
        elif event.button() == QtCore.Qt.LeftButton:
            frame = self.frame_at(pos.x())
            shift = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
            if event.modifiers() & QtCore.Qt.ControlModifier:
                start = run[0] if run is not None and len(run) == 1 else frame
                self.drag = {"kind": "range", "type": ctype, "first": start, "last": frame}
            elif c is not None:
                self.drag = {
                    "kind": "move",
                    "c": c,
                    "run": run,
                    "x": pos.x(),
                    "delta": 0,
                    "moved": False,
                    "ok": True,
                    "shift": shift,
                    "runs": [(c, run)],
                }
            else:
                self.drag = {"kind": "key", "type": ctype, "frame": frame, "start": pos, "shift": shift}
            self.update()

    def drag_track(self, drag, pos):
        if drag["kind"] == "range":
            drag["last"] = self.frame_at(pos.x())
            lo, hi = sorted((drag["first"], drag["last"]))
            self.set_hint(f"Range: frames {lo + 1} to {hi + 1}")
        elif drag["kind"] == "box":
            drag["end"] = pos
            self.marked = drag["base"] | self.runs_in(QtCore.QRectF(QtCore.QRect(drag["start"], pos).normalized()))
            self.set_hint(f"Select: {len(self.marked_runs())} marker{'' if len(self.marked_runs()) == 1 else 's'}")
        else:
            if not drag["moved"] and abs(pos.x() - drag["x"]) > 3:
                drag["moved"] = True
                if self.is_marked(drag["c"], drag["run"]):
                    drag["runs"] = self.marked_runs()
                else:
                    self.marked = {(drag["c"]["id"], f) for f in drag["run"]}
            if not drag["moved"]:
                return
            runs = [(c["id"], run) for c, run in drag["runs"]]
            shown = {"prompts": self.prompts, "constraints": self.constraints}
            delta, in_the_way = timeline.move_check(shown, runs, round((pos.x() - drag["x"]) / self.scale()))
            drag["delta"], drag["ok"] = delta, delta == 0 or not in_the_way
            where = (
                f"Move to frame {drag['run'][0] + delta + 1}"
                if len(runs) == 1
                else f"Move {len(runs)} markers by {delta:+d} frames"
            )
            self.set_hint(where + ("" if drag["ok"] else ": a constraint is in the way"))
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        drag = self.drag
        if (
            drag is not None
            and drag["kind"] == "key"
            and (pos - drag["start"]).manhattanLength() >= QtWidgets.QApplication.startDragDistance()
        ):
            drag.update(kind="box", base=set(self.marked) if drag["shift"] else set())
            self.drag_track(drag, pos)
        elif drag is not None and drag["kind"] in ("range", "move", "box"):
            self.drag_track(drag, pos)
        elif drag is not None and drag["kind"] == "scrub":
            self.frame_requested.emit(self.frame_at(pos.x()))
        elif drag is not None and drag["kind"] == "playhead":
            drag["moved"] = drag["moved"] or abs(pos.x() - drag["x"]) > 3
            if drag["moved"]:
                self.frame_requested.emit(self.frame_at(pos.x()))
        elif drag is None:
            row = self.row_at(pos)
            edge = row == 0 and self.prompt_at(pos)[1]
            self.setCursor(QtCore.Qt.SizeHorCursor if edge or self.on_playhead(pos) else QtCore.Qt.ArrowCursor)
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
        elif drag["kind"] == "playhead" and not drag["moved"] and drag["type"] is not None:
            self.marked = set()
            self.add_at_requested.emit(drag["type"], drag["frame"])
        elif drag["kind"] == "add" and self.row_at(pos) == 0 and self.prompt_at(pos)[0] is None:
            self.prompt_add_requested.emit()
        elif drag["kind"] == "box":
            pass
        elif drag["kind"] == "key" or (drag["kind"] == "range" and drag["first"] == drag["last"]):
            self.marked = set()
            self.add_at_requested.emit(drag["type"], drag["frame"] if drag["kind"] == "key" else drag["first"])
        elif drag["kind"] == "range":
            self.marked = set()
            self.range_requested.emit(drag["type"], drag["first"], drag["last"])
        elif drag["kind"] == "move" and not drag["moved"] and drag["shift"]:
            on = not self.is_marked(drag["c"], drag["run"])
            pairs = {(drag["c"]["id"], f) for f in drag["run"]}
            self.marked = self.marked | pairs if on else self.marked - pairs
        elif drag["kind"] == "move" and not drag["moved"]:
            self.marked = {(drag["c"]["id"], f) for f in drag["run"]}
            x = pos.x() / self.scale() - 0.5
            self.frame_requested.emit(min(drag["run"], key=lambda f: abs(f - x)))
        elif drag["kind"] == "move" and drag["delta"]:
            self.runs_move_requested.emit([(c["id"], run) for c, run in drag["runs"]], drag["delta"])
        self.update()
        self.mouseMoveEvent(event)
        if not self.shifting:
            self.selection_ended.emit()

    def mouseDoubleClickEvent(self, event):
        pos = event.position().toPoint()
        row = self.row_at(pos)
        pr, edge = self.prompt_at(pos) if row == 0 else (None, False)
        if pr is not None and not edge:
            self.edit_text(pr["id"])
        elif row and row >= 1 and event.button() == QtCore.Qt.LeftButton:
            if self.run_at(pos)[0] is not None:
                self.edit_requested.emit()

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
        QtWidgets.QApplication.instance().installEventFilter(self)

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
        QtWidgets.QApplication.instance().removeEventFilter(self)
        if editor is not None:
            editor.hide()
            editor.deleteLater()

    def eventFilter(self, obj, event):
        if (
            self.editor is not None
            and event.type() == QtCore.QEvent.MouseButtonPress
            and isinstance(obj, QtWidgets.QWidget)
            and obj is not self.editor
            and not self.editor.isAncestorOf(obj)
        ):
            self.finish_edit(self.editing)
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
        x = int(self.frame_x(self.frame))
        p.setPen(QtGui.QPen(QtGui.QColor(230, 90, 60), 2))
        p.drawLine(x, 0, x, self.height())
        for pr in prompts:
            self.paint_prompt(p, pr)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        waypoints = sorted(f for c in self.constraints if c["type"] == "root2d" for f in c["frames"])
        if self.smooth_path and len(waypoints) > 1:
            self.paint_constraint(p, {"type": "root2d"}, list(range(waypoints[0], waypoints[-1] + 1)), 70)
        drag = self.drag or {"kind": ""}
        moving = ({(c["id"], f) for c, run in drag["runs"] for f in run} if drag["kind"] == "move" and drag["moved"] else set())
        for c in self.constraints:
            staying = [f for f in c["frames"] if (c["id"], f) not in moving]
            for run in timeline.runs(staying):
                self.paint_constraint(p, c, run, marked=self.is_marked(c, run), changed=self.is_changed(c, run))
            landing = [f + drag["delta"] for f in c["frames"] if (c["id"], f) in moving]
            for run in timeline.runs(landing):
                self.paint_constraint(
                    p,
                    c,
                    run,
                    255 if drag["ok"] else 90,
                    marked=self.is_marked(c, [f - drag["delta"] for f in run]),
                    changed=bool(drag["delta"]) or self.is_changed(c, [f - drag["delta"] for f in run]),
                )
        if drag["kind"] == "range":
            lo, hi = sorted((drag["first"], drag["last"]))
            self.paint_constraint(p, {"type": drag["type"]}, list(range(lo, hi + 1)), 150)
        if drag["kind"] == "box":
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 170), 1))
            p.setBrush(QtGui.QColor(255, 255, 255, 30))
            p.drawRect(QtCore.QRectF(QtCore.QRect(drag["start"], drag["end"]).normalized()))
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
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
        p.fillRect(rect, QtGui.QColor(*CLIP_RGB))
        p.fillRect(rect.right() - 2, rect.top(), 3, rect.height(), QtGui.QColor(150, 190, 245))
        if pr["id"] == self.selected:
            width = outline_width()
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), width))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(QtCore.QRectF(rect).adjusted(width / 2, width / 2, -width / 2, -width / 2))
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

    def paint_constraint(self, p, c, frames=None, alpha=255, marked=False, changed=False):
        scale, y = self.scale(), self.track_y(c["type"])
        size = max(self.m.row_h // 2 - 2, 4)
        color = QtGui.QColor(*(motion_node.UNAPPLIED_MARKER_RGB if changed else motion_node.MARKER_RGB), alpha)
        outline = QtGui.QPen(QtGui.QColor(255, 255, 255, alpha), outline_width()) if marked else QtCore.Qt.NoPen
        p.setBrush(color)
        for run in timeline.runs(c["frames"] if frames is None else frames):
            p.setPen(outline)
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
        canvas.prompt_select_requested.connect(lambda pid: self.edit(timeline_edit.select_prompt, pid))
        canvas.add_at_requested.connect(lambda ctype, frame: self.with_node(motion_node.add_constraint, ctype, frame))
        canvas.runs_move_requested.connect(self.move_runs)
        canvas.selection_ended.connect(self.refresh)
        canvas.reset_requested.connect(lambda runs: self.edit(timeline_edit.reset_runs, runs))
        canvas.run_delete_requested.connect(lambda cid, run: self.edit(timeline_edit.delete_run, cid, run))
        canvas.range_requested.connect(lambda ctype, a, b: self.edit(timeline_edit.add_range, ctype, a, b))
        canvas.heading_requested.connect(lambda cid, on: self.edit(timeline_edit.set_heading, cid, on))
        canvas.hint_changed.connect(self.show_hover_hint)
        canvas.play_requested.connect(self.toggle_play)
        canvas.step_requested.connect(lambda step: hou.setFrame(max(int(round(hou.frame())) + step, 1)))
        self.reset_button = QtWidgets.QPushButton("Reset Selected")
        self.clear_button = QtWidgets.QPushButton("Clear All Constraints")
        self.reset_button.clicked.connect(self.reset_selected)
        self.clear_button.clicked.connect(lambda: self.edit(timeline_edit.clear_all))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.reset_button)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        self.button_rows.addLayout(row)
        self.place_buttons()
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
            if node is not None:
                node.path()
        except hou.ObjectWasDeleted:
            node = None
        self.timeline.setEnabled(node is not None)
        if node is None:
            self.status.show_hint("Select a Kimodo Motion node.")
        else:
            self.timeline.canvas.set_smooth_path(bool(node.evalParm("smooth_root_path")))
            d = self.doc
            canvas = self.timeline.canvas
            if d is not None and d["prompts"]:
                selected = d["prompts"][motion_node.selected_prompt(node, d)]["id"]
                if selected != canvas.selected:
                    canvas.selected = selected
                    canvas.update()
            canvas.resettable = motion_node.can_reset(node)
            self.reset_button.setEnabled(bool(canvas.marked) and canvas.can_reset(canvas.marked_runs()))
            if motion_node.is_editing(node) and not canvas.busy_selecting():
                posed = {(c["id"], f) for c in canvas.constraints for f in (timeline.run_at(c, canvas.frame) or ())}
                if posed != canvas.marked:
                    canvas.marked = posed
                    canvas.update()
            key = (
                node.parm("kimodo_doc").unexpandedString(),
                node.parm("kimodo_poses").unexpandedString(),
                node.evalParm("smooth_root_path"),
            )
            if key != self.guide_key:
                self.guide_key = key
                timeline_edit.update_root_path_guide(node)

    def place_buttons(self):
        super().place_buttons()
        if getattr(self, "reset_button", None) is not None:
            half = (self.generate_button.width() - 4) // 2
            for button in (self.reset_button, self.clear_button):
                if button.width() != half:
                    button.setFixedWidth(half)

    def reset_selected(self):
        runs = [(c["id"], run) for c, run in self.timeline.canvas.marked_runs()]
        if runs:
            self.edit(timeline_edit.reset_runs, runs)

    def show_hover_hint(self, text):
        self.hover_hint = text
        if self.node is not None:
            self.status.show_hint(text or motion_node.pose_hint(self.node))

    def on_playbar(self, event_type, frame):
        if event_type == hou.playbarEvent.FrameChanged:
            self.timeline.canvas.set_frame(int(round(frame)) - 1)

    def toggle_play(self):
        if hou.isUIAvailable():
            hou.playbar.stop() if hou.playbar.isPlaying() else hou.playbar.play()

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

    def move_runs(self, runs, delta):
        canvas = self.timeline.canvas
        before, key = canvas.marked, constraints_key(canvas.constraints)
        canvas.marked = canvas.moved_marks(runs, delta)
        if not self.edit(timeline_edit.move_runs, runs, delta):
            canvas.marked = before
            canvas.update()
        else:
            canvas.remember_move(key, before)

    def add_prompt(self):
        pid = self.edit(timeline_edit.add_prompt)
        if pid:
            self.timeline.canvas.edit_text(pid)
