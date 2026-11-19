"""pose state for kimodo motion: sidefx's apex animate state minus its ui."""

import hou

STATE_NAME = "kimodo_pose"
STATE_LABEL = "Kimodo Pose"
NEEDED_METHODS = (
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

        def updateHUD(self):
            self.scene_viewer.hudInfo(
                template={
                    "title": STATE_LABEL,
                    "icon": apexanimate.ANIMATE_TOOL_ICON,
                    "rows": [
                        {"id": "frame", "label": "Frame Controls", "key": su.hudHotkeyRef(self.hotkeys.frame_selected)},
                        {"id": "axissnapping", "label": "Enable axis snapping", "key": "Ctrl"},
                    ]
                    + apexanimate._buildHudGeneralRows(self.hotkeys),
                }
            )
            self.hud_initialized = True
            self._updateHUDHandleVis()

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
            result = super().onEnter(kwargs)
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
    from apex.xformgadget import registerXformHandle

    State = _state_class()
    category = hou.sopNodeTypeCategory()
    template = hou.ViewerStateTemplate(STATE_NAME, STATE_LABEL, category)
    hotkeys = asu.Hotkeys(STATE_NAME, category)
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
