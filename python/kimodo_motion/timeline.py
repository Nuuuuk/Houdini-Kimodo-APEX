"""timeline edits on the doc. pure functions, no hou."""

from . import doc

NEW_PROMPT_TEXT = "A person walks forward."
NEW_PROMPT_FRAMES = 5 * doc.FPS


def runs(frames):
    """[3, 7, 8, 9] -> [[3], [7, 8, 9]]"""
    out = []
    for f in frames:
        if out and f == out[-1][-1] + 1:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def run_at(constraint, frame):
    return next((r for r in runs(constraint["frames"]) if r[0] <= frame <= r[-1]), None)


def _new_id(d, key, prefix):
    used = {x["id"] for x in d[key]}
    return next(f"{prefix}{n}" for n in range(1, len(used) + 2) if f"{prefix}{n}" not in used)


def _find(d, key, item_id):
    for x in d[key]:
        if x["id"] == item_id:
            return x
    raise KeyError(item_id)


def _layout(d, lengths):
    """clips back to back with these lengths, constraint frames past the end dropped"""
    start = 0
    for p, n in zip(d["prompts"], lengths):
        p["start_frame"], p["end_frame"] = start, start + n
        start += n
    dropped, kept = [], []
    for c in d["constraints"]:
        dropped += [(c["type"], f) for f in c["frames"] if f >= start]
        c["frames"] = [f for f in c["frames"] if f < start]
        if c["frames"]:
            kept.append(c)
    d["constraints"] = kept
    return dropped


def _lengths(d):
    return [p["end_frame"] - p["start_frame"] for p in d["prompts"]]


def add_prompt(d, text=NEW_PROMPT_TEXT, frames=NEW_PROMPT_FRAMES):
    start = doc.total_frames(d)
    pid = _new_id(d, "prompts", "p")
    d["prompts"].append({"id": pid, "text": text, "start_frame": start, "end_frame": start + frames})
    return pid


def set_prompt_text(d, pid, text):
    _find(d, "prompts", pid)["text"] = text


def clamp_prompt_frames(frames):
    return min(max(int(frames), doc.MIN_PROMPT_FRAMES), doc.MAX_PROMPT_FRAMES)


def resize_prompt(d, pid, frames):
    lengths = [clamp_prompt_frames(frames) if p["id"] == pid else n for p, n in zip(d["prompts"], _lengths(d))]
    return _layout(d, lengths)


def delete_prompt(d, pid):
    if len(d["prompts"]) < 2:
        raise ValueError("the last prompt cannot be deleted")
    d["prompts"] = [p for p in d["prompts"] if p["id"] != _find(d, "prompts", pid)["id"]]
    return _layout(d, _lengths(d))


def _taken(d, ctype, skip=()):
    return {f for c in d["constraints"] if c["type"] == ctype for f in c["frames"] if (c["id"], f) not in skip}


def move_check(d, runs_, delta):
    """where a move would go: (delta, replaced)"""
    lo, hi = min(r[0] for _, r in runs_), max(r[-1] for _, r in runs_)
    delta = min(max(delta, -lo), doc.total_frames(d) - 1 - hi)
    moving = {(cid, f) for cid, run in runs_ for f in run}
    in_the_way = set()
    for cid, run in runs_:
        ctype = _find(d, "constraints", cid)["type"]
        in_the_way |= {(ctype, f) for f in _taken(d, ctype, skip=moving) & {f + delta for f in run}}
    return delta, sorted(in_the_way)


def move_runs(d, runs_, delta):
    delta, in_the_way = move_check(d, runs_, delta)
    if delta == 0 or in_the_way:
        return {}
    moved = {}
    for cid, run in runs_:
        moved.setdefault(cid, {}).update({f: f + delta for f in run})
    for cid, mapping in moved.items():
        c = _find(d, "constraints", cid)
        c["frames"] = sorted((set(c["frames"]) - set(mapping)) | set(mapping.values()))
    return moved


def add_range(d, ctype, first, last):
    lo, hi = sorted((first, last))
    lo, hi = max(lo, 0), min(hi, doc.total_frames(d) - 1)
    inside = set(range(lo, hi + 1))
    kept = []
    for c in d["constraints"]:
        if c["type"] == ctype:
            c["frames"] = [f for f in c["frames"] if f not in inside]
        if c["frames"]:
            kept.append(c)
    d["constraints"] = kept
    cid = _new_id(d, "constraints", "c")
    d["constraints"].append({"id": cid, "type": ctype, "frames": sorted(inside)})
    return cid


def add_keyframe(d, ctype, frame):
    if frame in _taken(d, ctype):
        return None
    cid = _new_id(d, "constraints", "c")
    d["constraints"].append({"id": cid, "type": ctype, "frames": [frame]})
    return cid


def delete_run(d, cid, run):
    c = _find(d, "constraints", cid)
    c["frames"] = [f for f in c["frames"] if f not in set(run)]
    if not c["frames"]:
        d["constraints"].remove(c)
