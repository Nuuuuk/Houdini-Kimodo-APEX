"""kimodo style example folders: meta.json and constraints.json."""

import glob
import json
import os
import urllib.parse
import urllib.request

from . import doc

OWN_KEY = "houdini"
OWN_SETTINGS = ("model", "postprocess", "root_margin", "smooth_root_path")
SOMA_EXAMPLES = os.path.join("assets", "demo", "examples", "kimodo-soma-rp")


def make_meta(prompts, settings):
    texts = [text for text, _ in prompts]
    durations = [frames / doc.FPS for _, frames in prompts]
    meta = {"text": texts[0], "duration": durations[0]} if len(texts) == 1 else {"texts": texts, "durations": durations}
    meta["num_samples"] = settings["num_samples"]
    meta["seed"] = settings["seed"]
    meta["diffusion_steps"] = settings["steps"]
    meta["cfg"] = {
        "enabled": True,
        "text_weight": settings["cfg_text"],
        "constraint_weight": settings["cfg_constraint"],
    }
    meta[OWN_KEY] = {name: settings[name] for name in OWN_SETTINGS}
    return meta


def _number(meta, key, kind=float):
    v = meta[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and v != int(v)):
        raise ValueError(f"meta.json: {key} must be {'a whole number' if kind is int else 'a number'}, not {v!r}")
    return kind(v)


def read_meta(meta):
    """(clips, settings, notes) from meta.json"""
    if not isinstance(meta, dict):
        raise ValueError("meta.json must hold a JSON object")
    if "text" in meta and "duration" in meta:
        texts, durations = [meta["text"]], [meta["duration"]]
    elif "texts" in meta and "durations" in meta:
        texts, durations = meta["texts"], meta["durations"]
        if not isinstance(texts, list) or not isinstance(durations, list) or len(texts) != len(durations) or not texts:
            raise ValueError("meta.json: texts and durations must be lists of the same length")
    else:
        raise ValueError("meta.json must have text and duration, or texts and durations")
    clips = []
    for n, (text, seconds) in enumerate(zip(texts, durations), 1):
        if not isinstance(text, str):
            raise ValueError(f"meta.json: the text of clip {n} must be a string")
        frames = max(1, int(round(_number({"duration": seconds}, "duration") * doc.FPS)))
        if not doc.MIN_PROMPT_FRAMES <= frames <= doc.MAX_PROMPT_FRAMES:
            raise ValueError(f"meta.json: clip {n} lasts {frames / doc.FPS:.2f} s; a clip lasts 2 to 10 s")
        clips.append((text, frames))
    settings, notes = {}, []
    for key, name, kind in (
        ("num_samples", "num_samples", int),
        ("seed", "seed", int),
        ("diffusion_steps", "steps", int),
    ):
        if key in meta:
            settings[name] = _number(meta, key, kind)
    cfg = meta.get("cfg", {})
    if not isinstance(cfg, dict):
        raise ValueError("meta.json: cfg must be an object")
    for key, name in (("text_weight", "cfg_text"), ("constraint_weight", "cfg_constraint")):
        if key in cfg:
            settings[name] = _number(cfg, key)
    if cfg.get("enabled", True) is False:
        notes.append("It turns CFG off, which the node cannot; its CFG weights are used.")
    own = meta.get(OWN_KEY, {})
    if not isinstance(own, dict):
        raise ValueError(f"meta.json: {OWN_KEY} must be an object")
    if "model" in own:
        if own["model"] not in doc.MODELS:
            raise ValueError(f"meta.json: unknown model {own['model']!r}")
        settings["model"] = own["model"]
    for name in ("postprocess", "smooth_root_path"):
        if name in own:
            if not isinstance(own[name], bool):
                raise ValueError(f"meta.json: {OWN_KEY}.{name} must be true or false")
            settings[name] = own[name]
    if "root_margin" in own:
        settings["root_margin"] = _number(own, "root_margin")
    return clips, settings, notes


def kimodo_package(root):
    """the kimodo package in the venv, None if missing"""
    sites = glob.glob(os.path.join(root, ".venv", "Lib", "site-packages")) + glob.glob(
        os.path.join(root, ".venv", "lib", "python*", "site-packages")
    )
    for site in sites:
        for info in glob.glob(os.path.join(site, "kimodo-*.dist-info")):
            try:
                with open(os.path.join(info, "direct_url.json")) as f:
                    url = json.load(f)
            except (OSError, ValueError):
                url = {}
            if url.get("dir_info", {}).get("editable") and url.get("url", "").startswith("file:"):
                package = os.path.join(urllib.request.url2pathname(urllib.parse.urlparse(url["url"]).path), "kimodo")
                if os.path.isdir(package):
                    return os.path.normpath(package)
        if os.path.isdir(os.path.join(site, "kimodo")):
            return os.path.join(site, "kimodo")
    return None


def soma_examples(root):
    package = kimodo_package(root)
    base = package and os.path.join(package, SOMA_EXAMPLES)
    if not base or not os.path.isdir(base):
        return {}
    names = sorted(n for n in os.listdir(base) if os.path.isfile(os.path.join(base, n, "meta.json")))
    return {n: os.path.join(base, n) for n in names}


def label(name):
    """'03_full_body_keyframes' -> '03 Full Body Keyframes'"""
    return " ".join(w if w.isdigit() else w.capitalize() for w in name.split("_"))
