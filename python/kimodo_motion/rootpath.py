"""smooth 2d root paths, same as the demo's make smooth path. no hou here."""

import math

import numpy as np

MARGIN = 0.06
MIN_WAYPOINTS_TO_SMOOTH = 3


def _smooth_level(target, start, margins, iters, alpha):
    n = len(target)
    a = np.zeros((n, n))
    for i in range(1, n - 1):
        a[i, i - 1 : i + 2] = (-1.0, 2.0, -1.0)
    m = a.T @ a
    step = 0.25 * math.sqrt(np.abs(np.diag(m)).max())
    solve = np.linalg.inv(m + step * np.eye(n))
    x, z, u = start.copy(), np.zeros_like(start), np.zeros_like(start)
    for _ in range(iters):
        z[:] = x + u - target
        norms = np.linalg.norm(z, axis=1)
        far = norms > margins
        if far.any():
            z[far] *= (margins[far] / norms[far])[:, None]
        z += target
        u += alpha * (x - z)
        x[:] = solve @ (step * (z - u))
    return x


def smooth_signal(x, margins, alpha=1.8, iters=500):
    """port of kimodo's smooth_signal, pos_weight 0"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    out[:] = x.mean(axis=0, keepdims=True)
    stepsize = 2 ** max(int(math.floor(math.log2(len(x)))) - 4, 1)
    while True:
        coarse = len(out[::stepsize])
        out[::stepsize] = _smooth_level(x[::stepsize], out[::stepsize], margins[::stepsize], iters, alpha)
        half = stepsize // 2
        between = len(out[half::stepsize])
        if between == coarse:
            out[half::stepsize][-1] = out[::stepsize][-1] + (out[::stepsize][-1] - out[::stepsize][-2]) / 2
            between -= 1
        out[half::stepsize][:between] = (out[::stepsize][:-1] + out[::stepsize][1:]) / 2
        if stepsize == 1:
            return out
        stepsize //= 2


_cache = {}


def dense_path(waypoints):
    """a point on every frame from the first waypoint to the last"""
    key = tuple(sorted((int(f), float(p[0]), float(p[1])) for f, p in waypoints.items()))
    if key not in _cache:
        at = np.array([k[0] for k in key])
        frames = np.arange(at[0], at[-1] + 1)
        xz = np.stack([np.interp(frames, at, [k[1] for k in key]), np.interp(frames, at, [k[2] for k in key])], axis=1)
        if len(key) >= MIN_WAYPOINTS_TO_SMOOTH:
            xz = smooth_signal(xz, np.full(len(xz), MARGIN))
        _cache.clear()
        _cache[key] = (frames.tolist(), xz)
    return _cache[key]


def waypoints_of(cjson):
    return {
        f: tuple(p) for e in cjson if e["type"] == "root2d" for f, p in zip(e["frame_indices"], e["smooth_root_2d"])
    }


def apply(cjson):
    """root2d keyframes -> one dense smoothed path"""
    waypoints = waypoints_of(cjson)
    if len(waypoints) < 2:
        return cjson
    frames, xz = dense_path(waypoints)
    first = frames[0]

    def on_path(frame_indices, fallback):
        return [
            xz[f - first].tolist() if first <= f <= frames[-1] else list(p) for f, p in zip(frame_indices, fallback)
        ]

    out = [
        {**e, "smooth_root_2d": on_path(e["frame_indices"], e["smooth_root_2d"])}
        for e in cjson
        if e["type"] != "root2d"
    ]
    out.append({"type": "root2d", "frame_indices": frames, "smooth_root_2d": xz.tolist()})
    out += [
        {**e, "smooth_root_2d": on_path(e["frame_indices"], e["smooth_root_2d"])}
        for e in cjson
        if e["type"] == "root2d" and "global_root_heading" in e
    ]
    return out


def guide(cjson):
    """what to draw in the viewport for the root2d entries"""
    lines = []
    for e in cjson:
        if e["type"] != "root2d":
            continue
        frames, xz = e["frame_indices"], e["smooth_root_2d"]
        start = 0
        for i in range(1, len(frames) + 1):
            if i == len(frames) or frames[i] != frames[i - 1] + 1:
                if i - start > 1:
                    lines.append([list(p) for p in xz[start:i]])
                start = i
    return {"lines": lines}
