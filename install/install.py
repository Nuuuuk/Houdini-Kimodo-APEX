"""Install kimodo motion for Houdini. Run it with any Python 3.10+, not hython."""

import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
PACKAGE_NAME = "kimodo_motion"


def find_hython(explicit=None):
    if explicit:
        return explicit
    exe = "hython.exe" if sys.platform == "win32" else "hython"
    if os.environ.get("HFS"):
        return os.path.join(os.environ["HFS"], "bin", exe)
    pattern = "C:/Program Files/Side Effects Software/Houdini 22.0.*" if sys.platform == "win32" else "/opt/hfs22.0*"
    builds = [p for p in glob.glob(pattern) if p.rsplit(".", 1)[-1].isdigit()]
    found = sorted(builds, key=lambda p: int(p.rsplit(".", 1)[-1]))
    if not found:
        sys.exit("Houdini 22.0 not found; pass --hython <path to hython>")
    return os.path.join(found[-1], "bin", exe)


def gui_env():
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    if sys.platform == "win32":
        env.pop("HOME", None)
    env.pop("HOUDINI_PACKAGE_DIR", None)
    return env


def run_hython(hython, code):
    out = subprocess.run([hython, "-c", code], env=gui_env(), capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1]


def write_pointer(pref_dir):
    path = os.path.join(pref_dir, "packages", f"{PACKAGE_NAME}.json")
    content = {"package_path": ROOT}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            old = f.read()
        try:
            if json.loads(old) == content:
                print(f"package pointer already up to date: {path}")
                return path
        except json.JSONDecodeError:
            pass
        print(f"replacing package pointer {path} (was: {old.strip()})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f)
        f.write("\n")
    print(f"wrote package pointer {path} -> {ROOT}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hython", help="path to Houdini 22.0's hython (default: $HFS or the newest 22.0 install)")
    args = ap.parse_args()

    hython = find_hython(args.hython)
    pref_dir = run_hython(hython, "import hou; print(hou.homeHoudiniDirectory())")
    print(f"Houdini user prefs: {pref_dir}")
    write_pointer(pref_dir)

    check = (
        "import hou, json; t = hou.sopNodeTypeCategory().nodeTypes().get('kimodo_motion'); "
        "print(json.dumps([hou.getenv('KIMODO_MOTION'), t is not None]))"
    )
    kimodo_motion, has_sop = json.loads(run_hython(hython, check))
    ok = kimodo_motion is not None and os.path.normcase(os.path.abspath(kimodo_motion)) == os.path.normcase(
        os.path.abspath(ROOT)
    )
    print(f"check: KIMODO_MOTION = {kimodo_motion}, Kimodo Motion SOP available: {has_sop}")
    if not (ok and has_sop):
        sys.exit("Houdini did not load the package; see README.md, Troubleshooting")
    print("kimodo_motion package installed.")


if __name__ == "__main__":
    main()
