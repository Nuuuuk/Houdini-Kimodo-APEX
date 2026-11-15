"""kimodo_doc, the json the node keeps its clips and constraints in."""

import copy
import json

VERSION = 1
FPS = 30
MODELS = ("Kimodo-SOMA-RP-v1.2", "Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-RP-v1")
CONSTRAINT_TYPES = ("fullbody", "root2d", "left-hand", "right-hand", "left-foot", "right-foot")
MIN_PROMPT_FRAMES = 2 * FPS
MAX_PROMPT_FRAMES = 10 * FPS

_DEFAULT = {
    "version": VERSION,
    "model": MODELS[0],
    "fps": FPS,
    "prompts": [{"id": "p1", "text": "A person walks forward.", "start_frame": 0, "end_frame": 5 * FPS}],
    "constraints": [],
    "generation": {
        "seed": 42,
        "steps": 100,
        "cfg_text": 2.0,
        "cfg_constraint": 2.0,
        "num_samples": 1,
        "postprocess": True,
        "root_margin": None,
    },
    "result": {"npz": [], "selected": 0},
}


class DocError(ValueError):
    pass


def new_doc():
    return copy.deepcopy(_DEFAULT)


def loads(text):
    """parse and validate, an empty string gives a new doc"""
    if not text or not text.strip():
        return new_doc()
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise DocError(f"kimodo_doc is not valid JSON: {e}") from None
    validate(doc)
    return doc


def dumps(doc):
    validate(doc)
    return json.dumps(doc, indent=1)


def total_frames(doc):
    return doc["prompts"][-1]["end_frame"] if doc["prompts"] else 0


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check(cond, msg):
    if not cond:
        raise DocError(msg)


def validate(doc):
    _check(isinstance(doc, dict), "document must be a JSON object")
    missing = set(_DEFAULT) - set(doc)
    _check(not missing, f"missing keys: {sorted(missing)}")
    _check(doc["version"] == VERSION, f"unsupported version {doc['version']!r}")
    _check(doc["model"] in MODELS, f"unknown model {doc['model']!r}")
    _check(doc["fps"] == FPS, f"fps must be {FPS}")

    prompts = doc["prompts"]
    _check(isinstance(prompts, list) and prompts, "prompts must be a non-empty list")
    expected_start = 0
    for p in prompts:
        _check(isinstance(p, dict) and isinstance(p.get("text"), str), f"bad prompt {p!r}")
        start, end = p.get("start_frame"), p.get("end_frame")
        _check(_is_int(start) and _is_int(end), f"prompt {p.get('id')}: frames must be integers")
        _check(start == expected_start, f"prompt {p.get('id')}: starts at {start}, expected {expected_start}")
        _check(
            MIN_PROMPT_FRAMES <= end - start <= MAX_PROMPT_FRAMES,
            f"prompt {p.get('id')}: {end - start} frames, must be {MIN_PROMPT_FRAMES}..{MAX_PROMPT_FRAMES}",
        )
        expected_start = end
    total = expected_start

    constraints = doc["constraints"]
    _check(isinstance(constraints, list), "constraints must be a list")
    for c in constraints:
        _check(isinstance(c, dict) and c.get("type") in CONSTRAINT_TYPES, f"bad constraint {c!r}")
        frames = c.get("frames")
        _check(
            isinstance(frames, list) and frames and all(_is_int(f) for f in frames),
            f"constraint {c.get('id')}: frames must be a non-empty list of integers",
        )
        _check(frames == sorted(set(frames)), f"constraint {c.get('id')}: frames must be sorted and unique")
        _check(0 <= frames[0] and frames[-1] < total, f"constraint {c.get('id')}: frames outside 0..{total - 1}")

    ids = [x.get("id") for x in prompts + constraints]
    _check(all(isinstance(i, str) and i for i in ids), "every prompt and constraint needs a string id")
    _check(len(ids) == len(set(ids)), "ids must be unique")

    g = doc["generation"]
    _check(isinstance(g, dict), "generation must be an object")
    _check(_is_int(g.get("seed")), "seed must be an integer")
    _check(_is_int(g.get("steps")) and g["steps"] > 0, "steps must be a positive integer")
    _check(_is_int(g.get("num_samples")) and g["num_samples"] > 0, "num_samples must be a positive integer")
    _check(_is_num(g.get("cfg_text")) and _is_num(g.get("cfg_constraint")), "cfg weights must be numbers")
    _check(isinstance(g.get("postprocess"), bool), "postprocess must be true or false")
    _check(g.get("root_margin") is None or _is_num(g["root_margin"]), "root_margin must be a number or null")

    r = doc["result"]
    _check(isinstance(r, dict) and isinstance(r.get("npz"), list), "result.npz must be a list")
    _check(all(isinstance(p, str) for p in r["npz"]), "result.npz entries must be paths")
    _check(_is_int(r.get("selected")), "result.selected must be an integer")
    _check(
        r["selected"] == 0 if not r["npz"] else 0 <= r["selected"] < len(r["npz"]),
        "result.selected is out of range",
    )
