"""talks to the kimodo service over http, stdlib only."""

import atexit
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HOST = "127.0.0.1"
PORT = 8765
SESSION_ID = uuid.uuid4().hex
_registered = set()


class ServiceError(RuntimeError):
    pass


def text_encoder_device():
    return os.environ.get("KIMODO_TEXT_ENCODER_DEVICE", "cuda")


def make_request(doc, constraints_json):
    """/generate body for a doc and its constraints json"""
    g = doc["generation"]
    return {
        "model": doc["model"],
        "prompts": [{"text": p["text"], "num_frames": p["end_frame"] - p["start_frame"]} for p in doc["prompts"]],
        "constraints": constraints_json,
        "seed": g["seed"],
        "steps": g["steps"],
        "cfg_text": g["cfg_text"],
        "cfg_constraint": g["cfg_constraint"],
        "num_samples": g["num_samples"],
        "postprocess": g["postprocess"],
        "root_margin": g["root_margin"],
        "text_encoder_device": text_encoder_device(),
    }


def cache_key(request):
    """hash of everything that changes the generated motion"""
    relevant = {k: v for k, v in request.items() if k not in ("output_dir", "bvh")}
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class Client:
    def __init__(self, host=HOST, port=PORT, timeout=10.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise ServiceError(f"{method} {path} failed with HTTP {e.code}: {detail}") from None
        return json.loads(raw) if raw else {}

    def health(self):
        try:
            return self._request("GET", "/health")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return None

    def generate(self, request):
        return self._request("POST", "/generate", request)["job_id"]

    def job(self, job_id):
        return self._request("GET", f"/jobs/{job_id}")

    def cancel(self, job_id):
        return self._request("POST", f"/jobs/{job_id}/cancel", {})

    def shutdown(self):
        return self._request("POST", "/shutdown", {})

    def register_session(self):
        if self.base in _registered:
            return
        self._request("POST", "/clients/register", {"id": SESSION_ID, "pid": os.getpid()})
        _registered.add(self.base)
        atexit.register(self._unregister_quietly)

    def _unregister_quietly(self):
        self.timeout = 2.0
        try:
            self._request("POST", "/clients/unregister", {"id": SESSION_ID})
        except Exception:
            pass

    def ensure_running(self, timeout=300.0, poll=1.0, on_wait=None):
        """start the service if needed and wait until it's ready"""
        health = self.health()
        proc = start_service() if health is None else None
        deadline = time.monotonic() + timeout
        while health is None or health.get("status") != "ready":
            check_exited(proc)
            if health is not None and health.get("status") == "error":
                raise ServiceError(f"Kimodo service failed to load: {health.get('error')} (see {log_path()})")
            if time.monotonic() > deadline:
                raise ServiceError(f"Kimodo service did not become ready within {timeout:.0f} s")
            if on_wait:
                on_wait(health)
            time.sleep(poll)
            health = self.health()
        return health


def service_command(root=None):
    root = root or os.environ.get("KIMODO_MOTION")
    if not root:
        raise ServiceError("KIMODO_MOTION is not set; is the kimodo_motion Houdini package loaded?")
    if sys.platform == "win32":
        python = os.path.join(root, ".venv", "Scripts", "python.exe")
    else:
        python = os.path.join(root, ".venv", "bin", "python")
    if not os.path.isfile(python):
        raise ServiceError(f"service venv not found at {python}; run the installer first")
    return [python, os.path.join(root, "service", "server.py")]


def check_exited(proc):
    """raise if a service we started has exited already"""
    if proc is not None and proc.poll() is not None:
        raise ServiceError(f"Kimodo service exited with code {proc.returncode}; see {log_path()}")


def log_path(root=None):
    return os.path.join(root or os.environ.get("KIMODO_MOTION", ""), "service.log")


def start_service(root=None):
    """starts the service detached, no window, output to service.log"""
    cmd = service_command(root)
    # houdini sets PYTHONHOME for its own python, the venv must not see it
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    log = open(log_path(root), "ab")
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT, "env": env}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kwargs)
    finally:
        log.close()
