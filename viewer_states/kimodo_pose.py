"""pose state for kimodo motion: sidefx's apex animate state minus its ui."""

import inspect
import math
import re

import hou

STATE_NAME = "kimodo_pose"
STATE_LABEL = "Kimodo Pose"
LABELS_KEY = "I"
LABEL_BOX_COLOR = (0.0, 0.0, 0.0, 0.55)
BOX_PAD = (9, 4)
UNPINNED_KEY = "U"
NEEDED_METHODS = (
    "onDraw",
    "onDrawInterrupt",
    "onExit",
    "updateHUD",
    "_updateHUDHandleVis",
    "_init_hud_widgets",
    "_loadHUDSettings",
    "_saveHUDSettings",
    "onEnter",
    "onKeyEvent",
    "onKeyTransitEvent",
    "onMouseEvent",
    "_frameSelectedControls",
    "writeToSopNode",
    "runSceneCallbacks",
    "_showControlsCmd",
    "_selectControlsCmd",
)
NEEDED_FUNCTIONS = ("_buildHudGeneralRows", "bindStateParms", "createViewerStateTemplate", "ANIMATE_TOOL_ICON")
NEEDED_ATTRIBUTES = (
    (
        "apexanimate",
        "State",
        (
            "scene_node",
            "scene_viewer",
            "hotkeys",
            "xform_gadget",
            "hud_initialized",
            "hud_parameter_panel",
            "hud_channel_panel",
            "is_shift_pressed",
            "is_ctrl_pressed",
            "scene",
            "scene_callbacks",
            "control_selection",
        ),
    ),
    ("apex.ui.settingshudwidgets", "AnimateSettingsWidget", ("widgethotkey", "hud_window")),
    ("apex.ui.splitchannelwidget", "SplitChannelWidgetHandler", ("widgethotkey", "hud_window", "is_visible")),
    ("apex.sceneevaluators", "SceneEvaluator", ("drawables",)),
    ("apex.scene_2", "SceneShapeOutputDrawable", ("drawable", "data_key", "output_path")),
)


def _sets(cls, name):
    """do the class's objects have this attribute"""
    found = re.compile(rf"\bself\.{name}\s*(:[^=\n]*)?=(?!=)")
    for c in cls.__mro__:
        if name in vars(c):
            return True
        try:
            if found.search(inspect.getsource(c)):
                return True
        except (OSError, TypeError):
            pass
    return False


def _missing():
    """what of sidefx's animate state this relies on and can't find"""
    import importlib

    try:
        import apex
        import apex.control_2 as control_2
        import apex.scene_2 as scene_2
        import apex.stateutils_2 as asu
        import apex.ui.statecommandutils as scu
        import apex.xformgadget as xformgadget
        import apex.xformgadgetutils as xgu
        import apexanimate
        import stateutils

        classes = [
            (f"{m}.{c}", getattr(importlib.import_module(m), c, None), names) for m, c, names in NEEDED_ATTRIBUTES
        ]
    except ImportError as e:
        return [str(e)]
    missing = [f"apexanimate.State.{name}" for name in NEEDED_METHODS if not hasattr(apexanimate.State, name)]
    missing += [f"apexanimate.{name}" for name in NEEDED_FUNCTIONS if not hasattr(apexanimate, name)]
    for module, name in (
        (asu, "Hotkeys"),
        (asu, "SUPPRESS_HOTKEYS"),
        (asu, "HK_RADIAL_MENU"),
        (scu, "registerAnimateState"),
        (xformgadget, "registerXformHandle"),
        (xgu, "isGlobalModeModifierMouseEvent"),
        (stateutils, "hudHotkeyRef"),
        (stateutils, "hotkeySymbolOrKeyString"),
        (stateutils, "defineHotkey"),
        (scene_2.SceneShapeOutputDrawable, "update"),
        (scene_2.Scene, "getGraphOutputData"),
        (scu, "getAnimateState"),
        (scene_2.Scene, "control_manager"),
        (scene_2.Scene, "getData"),
        (scene_2.Scene, "findDataPaths"),
        (getattr(apex, "ControlManager", None), "controls"),
        (getattr(apex, "Control", None), "node_properties"),
        (control_2, "getEnabledControlsFromTemplate"),
        (control_2, "setControlTemplateEnabled"),
    ):
        if not hasattr(module, name):
            missing.append(f"{getattr(module, '__name__', 'apex')}.{name}")
    for label, cls, names in classes:
        missing += [f"{label}.{name}" for name in names if cls is None or not _sets(cls, name)]
    return missing


def _state_class():
    import apex.scene_2 as scene_2
    import apex.stateutils_2 as asu
    import apex.xformgadgetutils as xgu
    import apexanimate
    import stateutils as su

    class State(apexanimate.State):
        _no_drag = False
        show_unpinned = True
        show_labels = True
        _label_text = None
        _label_boxes = None
        _title = STATE_LABEL
        _pose_lines = None
        _mark_lines = None
        _mark_key = None
        _pose_key = None
        _pelvis = None

        def updateHUD(self):
            unpinned_key = su.hudHotkeyRef(self.hotkeys.toggle_unpinned)
            labels_key = su.hudHotkeyRef(self.hotkeys.toggle_labels)
            self.scene_viewer.hudInfo(
                template={
                    "title": self._title,
                    "icon": apexanimate.ANIMATE_TOOL_ICON,
                    "rows": [
                        {"type": "divider", "label": "Posing"},
                        {"id": "frame", "label": "Frame Controls", "key": su.hudHotkeyRef(self.hotkeys.frame_selected)},
                        {"id": "axissnapping", "label": "Enable axis snapping", "key": "Ctrl"},
                    ]
                    + apexanimate._buildHudGeneralRows(self.hotkeys)
                    + [
                        {
                            "type": "group",
                            "id": "hideunpinnedgroup",
                            "rows": [{"label": "Hide Unpinned Controls", "key": unpinned_key}],
                        },
                        {
                            "type": "group",
                            "id": "showunpinnedgroup",
                            "rows": [{"label": "Show Unpinned Controls", "key": unpinned_key}],
                        },
                        {
                            "type": "group",
                            "id": "hidelabelsgroup",
                            "rows": [{"label": "Hide Pose Labels", "key": labels_key}],
                        },
                        {
                            "type": "group",
                            "id": "showlabelsgroup",
                            "rows": [{"label": "Show Pose Labels", "key": labels_key}],
                        },
                        {"id": "finish", "label": "End Posing", "key": "Esc"},
                    ],
                }
            )
            self.hud_initialized = True
            self._updateHUDHandleVis()
            self.scene_viewer.hudInfo(
                values={
                    "hideunpinnedgroup__visible": self.show_unpinned,
                    "showunpinnedgroup__visible": not self.show_unpinned,
                    "hidelabelsgroup__visible": self.show_labels,
                    "showlabelsgroup__visible": not self.show_labels,
                }
            )

        def _rig_skeletons(self):
            return [
                d
                for cb in self.scene_callbacks.values()
                for d in cb.drawables
                if isinstance(d, scene_2.SceneShapeOutputDrawable) and d.data_key.endswith(".skel")
            ]

        def _show_rig_skeleton(self, show):
            for skel in self._rig_skeletons():
                if skel.drawable.visible() != show:
                    skel.drawable.show(show)
                    if show:
                        skel.update(force=True)

        def _draw_pose(self, draw_handle, frame):
            from kimodo_motion import motion_node, rig, skeleton

            if self._pose_lines is None:
                return
            try:
                colors = motion_node.pose_colors(self.scene_node, frame)
                self._show_rig_skeleton(colors is None)
                if colors is None:
                    return
                skel = self._rig_skeletons()[0]
                rig_geo = self.scene.getGraphOutputData(skel.output_path, skel.data_key)
                node = self.scene_node
                key = (
                    frame,
                    rig_geo.pointFloatAttribValues("P"),
                    rig_geo.pointFloatAttribValues("transform"),
                    node.parm("kimodo_doc").unexpandedString(),
                    motion_node.shown_result(node),
                    tuple(colors),
                )
                if key != self._pose_key:
                    self._pose_key = key
                    positions = motion_node.posed_positions(node, frame, rig.pose_from_geometry(rig_geo))
                    self._pelvis = hou.Vector3(positions[0])
                    geo = hou.Geometry()
                    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
                    points = geo.createPoints(positions.tolist())
                    geo.createPolygons(
                        [(points[p], points[j]) for j, p in enumerate(skeleton.PARENTS) if p >= 0], False
                    )
                    geo.setPointFloatAttribValues("Cd", [v ** (1 / 2.2) for c in colors for v in c])
                    self._pose_lines.setGeometry(geo)
                view = self.scene_viewer.curViewport().viewTransform()
                eye = hou.Vector3(0.0, 0.0, 0.0) * view
                toward = (hou.Vector3(0.0, 0.0, 1.0) * view - eye).normalized()
                nearer = hou.hmath.buildTranslate(toward * (0.001 * (eye - self._pelvis).length()))
                self._pose_lines.setTransform(skel.drawable.transform() * nearer)
                self._pose_lines.draw(draw_handle)
            except Exception as e:
                self._pose_lines.show(False)
                self._pose_lines = None
                self._show_rig_skeleton(True)
                motion_node.set_status(
                    self.scene_node,
                    f"Kimodo could not draw the pose being edited ({type(e).__name__}: {e}); "
                    "the rig's own skeleton is shown.",
                )

        def _draw_mark(self, draw_handle, frame):
            from kimodo_motion import motion_node

            node = self.scene_node
            if self._mark_lines is None or frame not in motion_node._headed(node) or not self._rig_skeletons():
                return
            try:
                self._make_mark(draw_handle, frame)
            except Exception as e:
                self._mark_lines.show(False)
                self._mark_lines = None
                motion_node.set_status(node, f"Kimodo could not draw the heading's mark ({type(e).__name__}: {e}).")

        def _make_mark(self, draw_handle, frame):
            from kimodo_motion import motion_node, rig

            node = self.scene_node
            skel = self._rig_skeletons()[0]
            rig_geo = self.scene.getGraphOutputData(skel.output_path, skel.data_key)
            color = motion_node._marker_color(frame, motion_node.changed_frames(node))
            key = (frame, rig_geo.pointFloatAttribValues("P"), rig_geo.pointFloatAttribValues("transform"), color)
            if key != self._mark_key:
                self._mark_key = key
                geo = hou.Geometry()
                geo.addAttrib(hou.attribType.Point, "Cd", tuple(v ** (1 / 2.2) for v in color))
                for line in motion_node.posed_root_mark(node, frame, rig.pose_from_geometry(rig_geo)) or ():
                    geo.createPolygons([tuple(geo.createPoints(line))], False)
                self._mark_lines.setGeometry(geo)
            self._mark_lines.setTransform(skel.drawable.transform())
            self._mark_lines.draw(draw_handle)

        def onDraw(self, kwargs):
            super().onDraw(kwargs)
            self._draw_own(kwargs)

        def onDrawInterrupt(self, kwargs):
            super().onDrawInterrupt(kwargs)
            self._draw_own(kwargs)

        def _draw_own(self, kwargs):
            if self.scene_node is None:
                return
            current = int(round(hou.frame())) - 1
            self._draw_pose(kwargs["draw_handle"], current)
            self._draw_mark(kwargs["draw_handle"], current)
            try:
                from kimodo_motion import motion_node

                label = motion_node.constraint_label(self.scene_node, current)
                title = f"{STATE_LABEL}: {label}" if label else STATE_LABEL
                if title != self._title:
                    self._title = title
                    hou.ui.postEventCallback(self.updateHUD)
                if not self.show_labels or self._label_text is None:
                    return
                viewport = self.scene_viewer.curViewport()
                to_camera = viewport.viewTransform().inverted()
                boxes = hou.Geometry()
                texts = []
                for text, position, frame, color in motion_node.pose_labels(self.scene_node):
                    point = hou.Vector3(position)
                    if frame == current or (point * to_camera)[2] >= 0.0:
                        continue
                    at = viewport.mapToScreen(point)
                    width, height = self._label_text.size(text)
                    toward, _ = viewport.mapToWorld(at[0], at[1])

                    def on_plane(x, y):
                        direction, origin = viewport.mapToWorld(x, y)
                        return origin + direction * ((point - origin).dot(toward) / direction.dot(toward))

                    across = (on_plane(at[0] + 100, at[1]) - point) * 0.01
                    up = (on_plane(at[0], at[1] + 100) - point) * 0.01
                    w, h = width / 2.0 + BOX_PAD[0], height / 2.0 + BOX_PAD[1]
                    r = h * 0.6
                    corners = ((w - r, h - r, 0), (-w + r, h - r, 90), (-w + r, -h + r, 180), (w - r, -h + r, 270))
                    outline = [
                        (cx + r * math.cos(math.radians(a + s)), cy + r * math.sin(math.radians(a + s)))
                        for cx, cy, a in corners
                        for s in (0, 30, 60, 90)
                    ]
                    boxes.createPolygons([tuple(boxes.createPoints([point + across * x + up * y for x, y in outline]))])
                    shown = hou.Color(tuple(c ** (1 / 2.2) for c in color))
                    texts.append(
                        {
                            "text": text,
                            "color1": shown,
                            "origin": hou.drawableTextOrigin.BottomLeft,
                            "translate": hou.Vector3(at[0] - width / 2.0, at[1] - height * 0.75, 0.0),
                        }
                    )
                if self._label_boxes is not None:
                    self._label_boxes.setGeometry(boxes)
                    self._label_boxes.draw(kwargs["draw_handle"])
                for params in texts:
                    self._label_text.draw(kwargs["draw_handle"], params)
            except Exception:
                pass

        def onExit(self, kwargs):
            for lines in (self._pose_lines, self._mark_lines):
                if lines is not None:
                    lines.show(False)
            for drawable in (self._label_text, self._label_boxes):
                if drawable is not None:
                    drawable.show(False)
            return super().onExit(kwargs)

        def _toggle_unpinned(self):
            from kimodo_motion import motion_node

            self.show_unpinned = not self.show_unpinned
            self.updateHUD()
            try:
                motion_node.show_pinned_controls(self.scene_node, self.scene_viewer)
            except Exception:
                pass

        def _close_panels(self):
            for panel in (self.hud_parameter_panel, self.hud_channel_panel):
                if panel is not None:
                    panel.widgethotkey = ""
                    panel.hud_window.show(False)
            if self.hud_channel_panel is not None:
                self.hud_channel_panel.is_visible = False

        def _init_hud_widgets(self):
            super()._init_hud_widgets()
            self._close_panels()

        def _loadHUDSettings(self, *args, **kwargs):
            pass

        def _saveHUDSettings(self, *args, **kwargs):
            pass

        def onEnter(self, kwargs):
            self.show_unpinned = self.show_labels = True
            result = super().onEnter(kwargs)
            self._label_text = hou.TextDrawable(self.scene_viewer, "kimodo_pose_labels")
            self._label_text.show(True)
            self._pose_lines = hou.GeometryDrawable(
                self.scene_viewer,
                hou.drawableGeometryType.Line,
                "kimodo_pose",
                params={"use_cd": True, "line_width": 2.0},
            )
            self._pose_lines.show(True)
            self._mark_lines = hou.GeometryDrawable(
                self.scene_viewer,
                hou.drawableGeometryType.Line,
                "kimodo_root_mark",
                params={"use_cd": True, "line_width": 2.0},
            )
            self._mark_lines.show(True)
            self._mark_key = None
            self._pose_key = None
            self._title = STATE_LABEL
            self._label_boxes = hou.GeometryDrawable(
                self.scene_viewer,
                hou.drawableGeometryType.Face,
                "kimodo_label_boxes",
                params={"color1": LABEL_BOX_COLOR, "use_cd": False, "fade_factor": 1.0},
            )
            self._label_boxes.show(True)
            self._close_panels()
            for parm in kwargs.get("state_parms", {}).values():
                parm["visible"] = False
            return result

        def onKeyEvent(self, kwargs):
            if kwargs["ui_event"].device().isAutoRepeat():
                return False
            if self.xform_gadget.onKeyEvent(kwargs):
                return True
            key = su.hotkeySymbolOrKeyString(kwargs)
            if not key:
                return False
            if any(hou.hotkeys.isKeyMatch(key, symbol) for symbol in asu.SUPPRESS_HOTKEYS):
                return True
            if hou.hotkeys.isKeyMatch(key, asu.HK_RADIAL_MENU):
                return True
            if hou.hotkeys.isKeyMatch(key, self.hotkeys.frame_selected):
                self._frameSelectedControls()
                return True
            if hou.hotkeys.isKeyMatch(key, self.hotkeys.toggle_unpinned):
                self._toggle_unpinned()
                return True
            if hou.hotkeys.isKeyMatch(key, self.hotkeys.toggle_labels):
                self.show_labels = not self.show_labels
                self.updateHUD()
                self.scene_viewer.curViewport().draw()
                return True
            return False

        def onKeyTransitEvent(self, kwargs):
            device = kwargs["ui_event"].device()
            self.is_shift_pressed = device.isShiftKey()
            self.is_ctrl_pressed = device.isCtrlKey()
            return self.xform_gadget.onKeyTransitEvent(kwargs)

        def onMouseEvent(self, kwargs):
            reason = kwargs["ui_event"].reason()
            if reason == hou.uiEventReason.Start:
                self._no_drag = xgu.isGlobalModeModifierMouseEvent(kwargs)
            if self._no_drag and reason in (
                hou.uiEventReason.Start,
                hou.uiEventReason.Active,
                hou.uiEventReason.Changed,
            ):
                if reason == hou.uiEventReason.Changed:
                    self._no_drag = False
                return False
            return super().onMouseEvent(kwargs)

    return State


def createViewerStateTemplate():
    from kimodo_motion import motion_node

    missing = _missing()
    if missing:
        motion_node.POSE_STATE_PROBLEM = "this Houdini's animate state lacks " + ", ".join(missing)
        import apexanimate

        return apexanimate.createViewerStateTemplate(STATE_NAME, STATE_LABEL, apexanimate.State)

    import apex.stateutils_2 as asu
    import apex.ui.statecommandutils as scu
    import apexanimate
    import stateutils as su
    from apex.xformgadget import registerXformHandle

    State = _state_class()
    category = hou.sopNodeTypeCategory()
    template = hou.ViewerStateTemplate(STATE_NAME, STATE_LABEL, category)
    hotkeys = asu.Hotkeys(STATE_NAME, category)
    hotkeys.toggle_unpinned = su.defineHotkey(
        hotkeys.definitions(),
        STATE_NAME,
        "toggle_unpinned",
        UNPINNED_KEY,
        "Show or Hide Unpinned Controls",
        "Show dimmed, or hide, the controls that the frame's constraints do not pin",
        category,
    )
    hotkeys.toggle_labels = su.defineHotkey(
        hotkeys.definitions(),
        STATE_NAME,
        "toggle_labels",
        LABELS_KEY,
        "Show or Hide Pose Labels",
        "Show or hide the labels of the constraint poses",
        category,
    )
    template.bindFactory(lambda state_name, scene_viewer: State(state_name, scene_viewer, hotkeys))
    template.bindPlaybackChangeEvent()
    template.bindIcon(apexanimate.ANIMATE_TOOL_ICON)
    menu = hou.ViewerStateMenu("mainmenu", "Menu")
    registerXformHandle(template, main_menu=menu, main_hotkey_definitions=hotkeys.definitions())
    template.bindMenu(menu)
    template.bindHotkeyDefinitions(hotkeys.definitions())
    apexanimate.bindStateParms(template)
    scu.registerAnimateState(STATE_NAME)
    motion_node.POSE_STATE_PROBLEM = ""
    return template
