"""Install kimodo motion for Houdini. Run it with any Python 3.10+, not hython."""

import argparse
import ctypes
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
PACKAGE_NAME = "kimodo_motion"
DEPS = os.path.join(ROOT, ".deps")
UV = os.path.join(DEPS, "uv", "uv.exe")
VENV = os.path.join(ROOT, ".venv")
VENV_PYTHON = os.path.join(VENV, "Scripts", "python.exe")
REQUIREMENTS = "install/requirements.txt"
VENV_STATE = os.path.join(DEPS, "venv_lock.txt")
SETTINGS = os.path.join(ROOT, "settings.json")

# ungated copy of meta's repo, no hugging face account or token needed
LLAMA_REPO = "NousResearch/Meta-Llama-3-8B-Instruct"
LLAMA_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
LLAMA_SHARDS = {
    "model-00001-of-00004.safetensors": 4976698672,
    "model-00002-of-00004.safetensors": 4999802720,
    "model-00003-of-00004.safetensors": 4915916176,
    "model-00004-of-00004.safetensors": 1168138808,
}
LLAMA_NEEDED = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
LLAMA_FILES = (
    *LLAMA_SHARDS,
    *LLAMA_NEEDED,
    "generation_config.json",
    "special_tokens_map.json",
    "LICENSE",
    "USE_POLICY.md",
    "README.md",
)
LLAMA_LICENSE = "https://www.llama.com/llama3/license/"
# what the text encoder loads from these, the base model itself is the local llama
ADAPTERS = {
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp": (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised": ("adapter_config.json", "adapter_model.safetensors"),
}
KIMODO_MODEL = "nvidia/Kimodo-SOMA-RP-v1.2"

GB = 1024**3
# encoder and kimodo use ~15.3 GiB on the gpu, 16 GB cards report ~16300 MiB
GPU_ENCODER_MIN_VRAM_MIB = 15 * 1024
# on the cpu the encoder keeps llama in ram, the service peaked at 16 GiB
CPU_ENCODER_MIN_RAM = 30 * GB
# longest path inside is ~165 chars, windows allows 259 without long paths
MAX_ROOT_LENGTH = 90
VENV_BYTES = 6.5 * GB
LLAMA_BYTES = sum(LLAMA_SHARDS.values()) + 0.1 * GB
MODELS_BYTES = 1.5 * GB
# uninstall offers to delete the folder only when it holds nothing but these
TOOL_ENTRIES = (
    ".gitattributes",
    ".gitignore",
    "Install.bat",
    "Uninstall.bat",
    "README.md",
    "LICENSE",
    "NOTICE",
    "docs",
    "install",
    "kimodo_motion.json",
    "otls",
    "pytest.ini",
    "python",
    "python_panels",
    "service",
    "tests",
    "viewer_states",
)
INSTALLED_ENTRIES = (".deps", ".venv", "models", "settings.json", "service.log")
MODEL_ENTRIES = ("Meta-Llama-3-8B-Instruct", "hf")
KEEP_FOLDER = 3
DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


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
        sys.exit("Houdini 22.0 not found. Install Houdini 22.0 first, or pass --hython <path to hython>.")
    return os.path.join(found[-1], "bin", exe)


def gui_env():
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    if sys.platform == "win32":
        env.pop("HOME", None)
    env.pop("HOUDINI_PACKAGE_DIR", None)
    return env


def uv_env():
    """uv's cache and pythons inside .deps, no user config, kimodo built without compiling motion_correction"""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env.update(
        UV_CACHE_DIR=os.path.join(DEPS, "uv-cache"),
        UV_PYTHON_INSTALL_DIR=os.path.join(DEPS, "python"),
        UV_NO_CONFIG="1",
        SKIP_MOTION_CORRECTION_IN_SETUP="1",
    )
    return env


def venv_env():
    """env for .venv's python, with the hf cache the service uses"""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf"))
    env.update(
        # no telemetry, no xet chunk cache copy, no token or symlink nags
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_XET_CHUNK_CACHE_SIZE_BYTES="0",
        HF_HUB_VERBOSITY="error",
        HF_HUB_DISABLE_SYMLINKS_WARNING="1",
    )
    return env


def run_hython(hython, code):
    out = subprocess.run([hython, "-c", code], env=gui_env(), capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1]


def houdini_prefs(hython):
    """houdini's prefs folder, asked from houdini itself"""
    return json.loads(run_hython(hython, "import hou, json; print(json.dumps(hou.homeHoudiniDirectory()))"))


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def find_gpu():
    """first nvidia gpu from nvidia-smi, None if there's none"""
    exe = shutil.which("nvidia-smi") or os.path.join(
        os.environ.get("SystemRoot", "C:/Windows"), "System32", "nvidia-smi.exe"
    )
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit():
            return {"name": parts[0], "vram_mib": int(parts[1]), "driver": parts[2]}
    return None


def total_ram():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in ("phys", "avail_phys", "page", "avail_page", "virtual", "avail_virtual", "avail_extended")
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.phys


def encoder_device(gpu, ram):
    """(device, problem) for this gpu and ram"""
    if gpu["vram_mib"] >= GPU_ENCODER_MIN_VRAM_MIB:
        return "cuda", None
    if ram >= CPU_ENCODER_MIN_RAM:
        return "cpu", None
    return None, (
        f"Kimodo cannot run on this computer. Its GPU, {gpu['name']}, has {gpu['vram_mib'] / 1024:.0f} GB of "
        f"memory, under the 16 GB that Kimodo's text encoder needs on the GPU. The text encoder can run on the CPU "
        f"instead, but then it needs at least 32 GB of system memory (RAM), and this computer has {ram / GB:.0f} GB."
    )


def long_paths_enabled():
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def path_problem(root, long_paths):
    """a message if the folder's path is too long for windows"""
    if long_paths or len(root) <= MAX_ROOT_LENGTH:
        return None
    return (
        f"This folder's path is {len(root)} characters long: {root}\nWith the files inside it that is longer "
        f"than Windows allows. Move the folder to a shorter path (at most {MAX_ROOT_LENGTH} characters), for "
        f"example C:\\kimodo_motion, and run Install.bat there."
    )


def existing_parent(path):
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return path


def disk_problem(needs, free=lambda path: shutil.disk_usage(path).free):
    """a message if a drive is short on space"""
    per_drive = {}
    for path, size in needs:
        drive = existing_parent(os.path.abspath(path))
        key = os.path.splitdrive(drive)[0].upper() or drive
        per_drive.setdefault(key, [drive, 0])[1] += size
    for key, (path, size) in per_drive.items():
        if size and free(path) < size + GB:
            return (
                f"Not enough free disk space on {key}: the install needs about {size / GB:.0f} GB there and "
                f"{free(path) / GB:.0f} GB is free. Free some space, or move this folder to another drive, and "
                f"run Install.bat again."
            )
    return None


def msvc_runtime_present():
    try:
        ctypes.WinDLL("msvcp140.dll")
        return True
    except OSError:
        return False


def install_msvc_runtime():
    print(
        "Microsoft's Visual C++ runtime, which PyTorch needs, is missing. Installing it with Microsoft's own "
        "installer; Windows will ask you to allow it.",
        flush=True,
    )
    exe = os.path.join(DEPS, "vc_redist.x64.exe")
    os.makedirs(DEPS, exist_ok=True)
    subprocess.run(["curl.exe", "-fL", "-#", "--retry", "3", "-o", exe, VC_REDIST_URL], check=True)
    quoted = exe.replace("'", "''")
    check = (
        f"$s = Get-AuthenticodeSignature -LiteralPath '{quoted}'; if ($s.Status -eq 'Valid' -and "
        "$s.SignerCertificate.Subject -like '*O=Microsoft Corporation*') { exit 0 }; exit 1"
    )
    signed = subprocess.run(["powershell", "-NoProfile", "-Command", check])
    if signed.returncode != 0:
        sys.exit("The downloaded Visual C++ installer is not signed by Microsoft; it was not run.")
    code = subprocess.run([exe, "/install", "/passive", "/norestart"]).returncode
    os.remove(exe)
    if code not in (0, 1638, 3010) or not msvc_runtime_present():
        sys.exit(
            f"Microsoft's Visual C++ runtime was not installed (installer exit code {code}). Install it from "
            f"{VC_REDIST_URL} and run Install.bat again."
        )


def check_torch_cuda(gpu):
    code = "import torch; t = torch.ones(8, device='cuda'); print(torch.cuda.get_device_name(0), float(t.sum()))"
    out = subprocess.run([VENV_PYTHON, "-X", "utf8", "-c", code], env=venv_env(), capture_output=True, text=True)
    if out.returncode != 0:
        error = (out.stderr.strip().splitlines() or ["no output"])[-1]
        sys.exit(
            f"PyTorch could not use the GPU: {error}\nThis usually means the NVIDIA driver (version "
            f"{gpu['driver']}) is too old for CUDA 12.8. Install the latest driver from {DRIVER_URL}, restart "
            f"the computer, and run Install.bat again."
        )


def venv_action(venv, state, lock_hash):
    """'make', 'sync', 'done' or 'foreign' (a .venv we didn't make)"""
    if state is None and os.path.exists(venv):
        return "foreign"
    if not os.path.isfile(os.path.join(venv, "Scripts", "python.exe")):
        return "make"
    return "done" if state == lock_hash else "sync"


def make_venv(action, lock_hash):
    if action in ("done", "foreign"):
        return
    if not os.path.isfile(UV):
        sys.exit(f"uv not found at {UV}; run Install.bat.")
    print("Python packages (5 GB), this takes a few minutes", flush=True)
    write_text(VENV_STATE, "pending")
    if action == "make":
        subprocess.run(
            [UV, "venv", VENV, "--python", sys.executable, "--allow-existing", "--no-project", "-q"],
            cwd=ROOT,
            env=uv_env(),
            check=True,
        )
    subprocess.run(
        [UV, "pip", "sync", REQUIREMENTS, "--python", VENV_PYTHON, "--strict", "-q"], cwd=ROOT, env=uv_env(), check=True
    )
    write_text(VENV_STATE, lock_hash)
    # the cache hard links all of .venv and explorer counts it twice, so drop it once installed
    subprocess.run([UV, "cache", "clean", "-q"], cwd=ROOT, env=uv_env())


def llama_dir():
    path = os.environ.get("KIMODO_LLAMA_DIR") or os.path.join(ROOT, "models", "Meta-Llama-3-8B-Instruct")
    return os.path.normpath(path)


def llama_present(path):
    """a complete llama copy is there"""
    if not all(os.path.isfile(os.path.join(path, name)) for name in LLAMA_NEEDED):
        return False
    return all(
        os.path.isfile(os.path.join(path, name)) and os.path.getsize(os.path.join(path, name)) == size
        for name, size in LLAMA_SHARDS.items()
    )


def models_cached():
    """adapters and kimodo checkpoint are in the hf cache"""
    hub = os.path.join(venv_env()["HF_HOME"], "hub")
    repos = (*ADAPTERS, KIMODO_MODEL)
    return all(os.path.isdir(os.path.join(hub, "models--" + repo.replace("/", "--"))) for repo in repos)


def download_llama(path):
    """curl each file into .part, resuming, then rename it"""
    print(f"Meta-Llama-3-8B-Instruct ({LLAMA_BYTES / GB:.0f} GB)", flush=True)
    os.makedirs(path, exist_ok=True)
    for name in LLAMA_FILES:
        target, size = os.path.join(path, name), LLAMA_SHARDS.get(name)
        if os.path.isfile(target) and (size is None or os.path.getsize(target) == size):
            continue
        part = target + ".part"
        if size is None or not os.path.isfile(part) or os.path.getsize(part) != size:
            url = f"https://huggingface.co/{LLAMA_REPO}/resolve/{LLAMA_REVISION}/{name}"
            # shards resume and show a bar, the small files come whole and quiet
            how = ["-#", "-C", "-"] if size else ["-sS"]
            subprocess.run(["curl.exe", "-fL", "--retry", "5", "--retry-delay", "5", *how, "-o", part, url], check=True)
        if size is not None and os.path.getsize(part) != size:
            sys.exit(f"{name} has {os.path.getsize(part)} bytes instead of {size}; run Install.bat again.")
        os.replace(part, target)


def fetch_models(llama, have_llama):
    if not have_llama:
        download_llama(llama)
        if not llama_present(llama):
            sys.exit(f"Meta-Llama-3-8B-Instruct in {llama} is still incomplete; run Install.bat again.")
    script = os.path.join(ROOT, "install", "fetch_models.py")
    subprocess.run([VENV_PYTHON, "-X", "utf8", script], env=venv_env(), check=True)


def write_settings(device, path=SETTINGS):
    """text encoder device into settings.json, nothing for the default cuda"""
    try:
        with open(path, encoding="utf-8") as f:
            settings = json.load(f)
    except (OSError, ValueError):
        settings = {}
    if settings.get("text_encoder_device", "cuda") == device:
        return
    settings["text_encoder_device"] = device
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)
        f.write("\n")


def write_pointer(pref_dir):
    path = os.path.join(pref_dir, "packages", f"{PACKAGE_NAME}.json")
    content = {"package_path": ROOT}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            old = f.read()
        try:
            if json.loads(old) == content:
                return path
        except json.JSONDecodeError:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f)
        f.write("\n")
    return path


def remove_pointer(pref_dir):
    """remove the package pointer if it points here"""
    path = os.path.join(pref_dir, "packages", f"{PACKAGE_NAME}.json")
    try:
        with open(path, encoding="utf-8") as f:
            target = json.load(f).get("package_path", "")
    except OSError:
        print(f"No package pointer in {os.path.dirname(path)}; Houdini is not linked to Kimodo Motion.")
        return
    except (ValueError, AttributeError):
        target = ""
    if os.path.normcase(os.path.abspath(target or "?")) != os.path.normcase(os.path.abspath(ROOT)):
        print(f"The package pointer {path} points at another folder ({target}); it is left as it is.")
        return
    os.remove(path)
    print(f"Removed the package pointer {path}: Houdini no longer loads Kimodo Motion.")


def foreign_entries(root):
    """what's in the folder that neither the zip nor Install.bat put there"""
    own = {name.lower() for name in TOOL_ENTRIES + INSTALLED_ENTRIES}
    found = [name for name in os.listdir(root) if name.lower() not in own]
    models = os.path.join(root, "models")
    if os.path.isdir(models):
        own_models = {name.lower() for name in MODEL_ENTRIES}
        found += [os.path.join("models", name) for name in os.listdir(models) if name.lower() not in own_models]
    return sorted(found, key=str.lower)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hython", help="path to Houdini 22.0's hython (default: $HFS or the newest 22.0 install)")
    ap.add_argument("--uninstall", action="store_true", help="remove the package pointer to this folder")
    args = ap.parse_args()

    hython = find_hython(args.hython)
    if args.uninstall:
        remove_pointer(houdini_prefs(hython))
        others = foreign_entries(ROOT)
        if others:
            print(
                f"\n{os.path.normpath(ROOT)} holds things that Install.bat did not put there, so it is not offered "
                "for deletion:"
            )
            for name in others[:20]:
                print("  " + name)
            if len(others) > 20:
                print(f"  and {len(others) - 20} more")
            sys.exit(KEEP_FOLDER)
        return
    gpu = find_gpu()
    if gpu is None:
        sys.exit(f"No NVIDIA GPU found. Kimodo needs an NVIDIA GPU with its driver installed ({DRIVER_URL}).")
    device, problem = encoder_device(gpu, total_ram())
    problem = problem or path_problem(ROOT, long_paths_enabled())
    lock_hash = file_hash(os.path.join(ROOT, REQUIREMENTS))
    action = venv_action(VENV, read_text(VENV_STATE), lock_hash)
    llama = llama_dir()
    have_llama = llama_present(llama)
    problem = problem or disk_problem(
        [
            (ROOT, VENV_BYTES if action in ("make", "sync") else 0),
            (llama, 0 if have_llama else LLAMA_BYTES),
            (ROOT, 0 if models_cached() else MODELS_BYTES),
        ]
    )
    if problem:
        sys.exit(problem)
    if not msvc_runtime_present():
        install_msvc_runtime()

    make_venv(action, lock_hash)
    check_torch_cuda(gpu)
    fetch_models(llama, have_llama)
    write_settings(device)

    write_pointer(houdini_prefs(hython))

    check = (
        "import hou, json; t = hou.sopNodeTypeCategory().nodeTypes().get('kimodo_motion'); "
        "print(json.dumps([hou.getenv('KIMODO_MOTION'), t is not None]))"
    )
    kimodo_motion, has_sop = json.loads(run_hython(hython, check))
    ok = kimodo_motion is not None and os.path.normcase(os.path.abspath(kimodo_motion)) == os.path.normcase(
        os.path.abspath(ROOT)
    )
    if not (ok and has_sop):
        sys.exit(
            f"Houdini did not load the package (KIMODO_MOTION = {kimodo_motion}, Kimodo Motion SOP: {has_sop}); "
            "see README.md, Troubleshooting"
        )
    print()
    print("Kimodo Motion is installed.")
    print(f"Using it means agreeing to the Meta Llama 3 Community License: {LLAMA_LICENSE}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"{os.path.basename(str(e.cmd[0]))} stopped with exit code {e.returncode}; the reason is above.")
