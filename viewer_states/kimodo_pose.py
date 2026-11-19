"""pose state for kimodo motion: sidefx's apex animate state minus its ui."""

import hou

STATE_NAME = "kimodo_pose"
STATE_LABEL = "Kimodo Pose"
LABELS_KEY = "I"
LABEL_BOX_COLOR = (0.0, 0.0, 0.0, 0.55)
BOX_PAD = (9, 4)
LABEL_BOXES = 48
UNPINNED_KEY = "U"
NEEDED_METHODS = (
    "onDraw",
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
)
NEEDED_FUNCTIONS = ("_buildHudGeneralRows", "bindStateParms", "createViewerStateTemplate", "ANIMATE_TOOL_ICON")


def _missing():
    """what of sidefx's animate state this relies on and can't find"""
    try:
        import apex.stateutils_2 as asu
        import apex.ui.statecommandutils as scu
        import apex.xformgadget as xformgadget
        import apex.xformgadgetutils as xgu
        import apexanimate
        import stateutils
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
    ):
        if not hasattr(module, name):
            missing.append(f"{module.__name__}.{name}")
    return missing


def _state_class():
    import apex.stateutils_2 as asu
    import apex.xformgadgetutils as xgu
    import apexanimate
    import stateutils as su

    class State(apexanimate.State):
        _no_drag = False
        show_unpinned = True
        show_labels = True
        _label_text = None
        _label_boxes = ()
        _title = STATE_LABEL

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

        def onDraw(self, kwargs):
            super().onDraw(kwargs)
            if self.scene_node is None:
                return
            try:
                from kimodo_motion import motion_node

                current = int(round(hou.frame())) - 1
                label = motion_node.constraint_label(self.scene_node, current)
                title = f"{STATE_LABEL}: {label}" if label else STATE_LABEL
                if title != self._title:
                    self._title = title
                    hou.ui.postEventCallback(self.updateHUD)
                if not self.show_labels or self._label_text is None:
                    return
                viewport = self.scene_viewer.curViewport()
                to_camera = viewport.viewTransform().inverted()
                boxes = iter(self._label_boxes)
                for text, position, frame, color in motion_node.pose_labels(self.scene_node):
                    point = hou.Vector3(position)
                    if frame == current or (point * to_camera)[2] >= 0.0:
                        continue
                    at = viewport.mapToScreen(point)
                    width, height = self._label_text.size(text)
                    box = next(boxes, None)
                    if box is not None:
                        toward, _ = viewport.mapToWorld(at[0], at[1])

                        def on_plane(x, y):
                            direction, origin = viewport.mapToWorld(x, y)
                            return origin + direction * ((point - origin).dot(toward) / direction.dot(toward))

                        across, along = on_plane(at[0] + 100, at[1]) - point, on_plane(at[0], at[1] + 100) - point
                        meter = across.length() / 100.0
                        x, y = across.normalized(), along.normalized()
                        z = x.cross(y)
                        box.setTransform(
                            hou.Matrix4(
                                (
                                    (x[0], x[1], x[2], 0),
                                    (y[0], y[1], y[2], 0),
                                    (z[0], z[1], z[2], 0),
                                    (point[0], point[1], point[2], 1),
                                )
                            )
                        )
                        w, h = (width + 2 * BOX_PAD[0]) * meter, (height + 2 * BOX_PAD[1]) * meter
                        box.setParams(
                            {
                                "size": (w, h),
                                "position": (0.0, 0.0),
                                "radius": h * 0.3,
                                "fill": True,
                                "color": LABEL_BOX_COLOR,
                            }
                        )
                        box.draw(kwargs["draw_handle"])
                    shown = hou.Color(tuple(c ** (1 / 2.2) for c in color))
                    self._label_text.draw(
                        kwargs["draw_handle"],
                        {
                            "text": text,
                            "color1": shown,
                            "origin": hou.drawableTextOrigin.BottomLeft,
                            "translate": hou.Vector3(at[0] - width / 2.0, at[1] - height * 0.75, 0.0),
                        },
                    )
            except Exception:
                pass

        def onExit(self, kwargs):
            if self._label_text is not None:
                self._label_text.show(False)
            for box in self._label_boxes:
                box.show(False)
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
            self._title = STATE_LABEL
            try:
                self._label_boxes = [
                    hou.Drawable2D(self.scene_viewer, hou.drawable2DType.Rect, f"kimodo_label_box{i}", pickable=False)
                    for i in range(LABEL_BOXES)
                ]
                for box in self._label_boxes:
                    box.show(True)
            except Exception:
                self._label_boxes = ()
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
