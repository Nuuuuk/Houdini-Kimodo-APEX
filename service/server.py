"""Kimodo inference service. Listens on 127.0.0.1 only."""

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SERVICE_NAME = "kimodo_motion"
SERVICE_VERSION = 2
DEFAULT_MODEL = "Kimodo-SOMA-RP-v1.2"
MODELS = ("Kimodo-SOMA-RP-v1.2", "Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-RP-v1")
NUM_TRANSITION_FRAMES = 5


def default_text_encoder_device():
    """env KIMODO_TEXT_ENCODER_DEVICE, else settings.json, else cuda"""
    value = os.environ.get("KIMODO_TEXT_ENCODER_DEVICE")
    if value:
        return value
    try:
        with open(os.path.join(ROOT, "settings.json"), encoding="utf-8") as f:
            return json.load(f).get("text_encoder_device", "cuda")
    except (OSError, ValueError, AttributeError):
        return "cuda"


def set_default_env():
    os.environ.setdefault("KIMODO_LLAMA_DIR", os.path.join(ROOT, "models", "Meta-Llama-3-8B-Instruct"))
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf"))
    os.environ.setdefault("LOCAL_CACHE", "true")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    # the fortran runtime scipy pulls in kills the process on console close otherwise
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")


class RequestError(ValueError):
    pass


def parse_request(body):
    """validate a /generate body and fill in defaults"""

    def need(cond, msg):
        if not cond:
            raise RequestError(msg)

    need(isinstance(body, dict), "request must be a JSON object")
    model = body.get("model", DEFAULT_MODEL)
    need(model in MODELS, f"unknown model {model!r}")
    prompts = body.get("prompts")
    need(isinstance(prompts, list) and prompts, "prompts must be a non-empty list")
    for p in prompts:
        need(isinstance(p, dict) and isinstance(p.get("text"), str), f"bad prompt {p!r}")
        need(isinstance(p.get("num_frames"), int) and p["num_frames"] > 0, f"bad num_frames in {p!r}")
    constraints = body.get("constraints", [])
    need(isinstance(constraints, list), "constraints must be a list (Kimodo constraints JSON)")
    out_dir = body.get("output_dir")
    need(isinstance(out_dir, str) and os.path.isabs(out_dir), "output_dir must be an absolute path")
    req = {
        "model": model,
        "texts": [p["text"] for p in prompts],
        "num_frames": [p["num_frames"] for p in prompts],
        "constraints": constraints,
        "seed": body.get("seed", 42),
        "steps": body.get("steps", 100),
        "cfg_text": float(body.get("cfg_text", 2.0)),
        "cfg_constraint": float(body.get("cfg_constraint", 2.0)),
        "num_samples": body.get("num_samples", 1),
        "postprocess": bool(body.get("postprocess", True)),
        "root_margin": body.get("root_margin"),
        "first_heading_angle": body.get("first_heading_angle"),
        "output_dir": out_dir,
        "bvh": bool(body.get("bvh", False)),
    }
    need(isinstance(req["seed"], int), "seed must be an integer")
    need(isinstance(req["steps"], int) and req["steps"] > 0, "steps must be a positive integer")
    need(isinstance(req["num_samples"], int) and req["num_samples"] > 0, "num_samples must be a positive integer")
    need(req["root_margin"] is None or isinstance(req["root_margin"], (int, float)), "root_margin must be a number")
    need(
        req["first_heading_angle"] is None or isinstance(req["first_heading_angle"], (int, float)),
        "first_heading_angle must be a number",
    )
    req["text_encoder_device"] = body.get("text_encoder_device")
    need(req["text_encoder_device"] in (None, "cuda", "cpu"), "text_encoder_device must be cuda or cpu")
    return req


class CachedTextEncoder:
    """per prompt cache in front of kimodo's text encoder"""

    def __init__(self, encoder, device):
        import torch

        self.encoder, self.device, self.cache = encoder, device, {}
        self._torch = torch

    def __call__(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            feats, _ = self.encoder(missing)
            for t, f in zip(missing, feats):
                self.cache[t] = f.detach().to(self.device)
        feat = self._torch.stack([self.cache[t] for t in texts]).clone()
        lengths = [1] * len(texts)
        return (feat[0], lengths[0]) if single else (feat, lengths)

    def to(self, device):
        return self

    def __getattr__(self, name):
        return getattr(self.encoder, name)


class Canceled(Exception):
    pass


def pid_alive(pid):
    import psutil

    return psutil.pid_exists(pid)


class Service:
    def __init__(self, text_encoder_device="cuda", preload=(DEFAULT_MODEL,), idle_timeout=30 * 60.0):
        self.text_encoder_device = text_encoder_device
        self.preload = preload
        self.status, self.error = "loading", None
        self.models, self.text_encoder = {}, None
        self.jobs, self.queue = {}, queue.Queue()
        self.lock = threading.Lock()
        self.idle_timeout = idle_timeout
        self.clients, self.had_clients = {}, False
        self.last_activity = time.monotonic()
        self.pid_alive = pid_alive

    def register(self, client_id, pid):
        with self.lock:
            self.clients[client_id] = int(pid)
            self.had_clients = True
            self.last_activity = time.monotonic()
        print(f"client registered: {client_id} (pid {pid}), {len(self.clients)} active", flush=True)

    def unregister(self, client_id):
        with self.lock:
            self.clients.pop(client_id, None)
        print(f"client unregistered: {client_id}, {len(self.clients)} active", flush=True)

    def busy(self):
        return any(j["status"] in ("queued", "running") for j in self.jobs.values())

    def check_lifetime(self, now=None):
        """reason to shut down now, or None"""
        now = time.monotonic() if now is None else now
        with self.lock:
            for cid, pid in list(self.clients.items()):
                if not self.pid_alive(pid):
                    print(f"client {cid} (pid {pid}) is gone", flush=True)
                    del self.clients[cid]
            if self.busy():
                return None
            if self.had_clients and not self.clients:
                return "the last Houdini session using the service has exited"
            if self.idle_timeout and now - self.last_activity > self.idle_timeout:
                return f"idle for {self.idle_timeout / 60:.0f} min"
        return None

    def monitor(self, stop, interval=5.0):
        while True:
            time.sleep(interval)
            reason = self.check_lifetime()
            if reason:
                print(f"Shutting down: {reason}", flush=True)
                stop()
                return

    def load(self):
        try:
            import kimodo_patches

            kimodo_patches.apply()
            for name in self.preload:
                self.get_model(name)
            self.status = "ready"
            print("Kimodo service ready", flush=True)
        except Exception as e:
            traceback.print_exc()
            self.status, self.error = "error", f"{type(e).__name__}: {e}"

    def get_model(self, name):
        if name not in self.models:
            import kimodo

            print(f"Loading {name} ...", flush=True)
            if self.text_encoder is None:
                model = kimodo.load_model(name, device="cuda:0")
                self.text_encoder = CachedTextEncoder(model.text_encoder, "cuda:0")
            else:
                model = kimodo.load_model(name, device="cuda:0", text_encoder=self.text_encoder)
            model.text_encoder = self.text_encoder
            self.models[name] = model
        return self.models[name]

    def health(self):
        vram = None
        # don't touch torch while the loader thread is still importing it
        if self.status == "ready":
            import torch

            vram = round(torch.cuda.memory_allocated() / 2**30, 2)
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "status": self.status,
            "error": self.error,
            "models": sorted(self.models),
            "vram_gib": vram,
            "text_encoder_device": self.text_encoder_device,
            "cached_prompts": len(self.text_encoder.cache) if self.text_encoder else 0,
            "clients": len(self.clients),
            "idle_timeout_min": (self.idle_timeout or 0) / 60,
        }

    def submit(self, body):
        req = parse_request(body)
        if req["text_encoder_device"] not in (None, self.text_encoder_device):
            raise RequestError(
                f"the running service encodes text on {self.text_encoder_device}, the request expects "
                f"{req['text_encoder_device']}; shut the service down so it restarts with the new setting"
            )
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.last_activity = time.monotonic()
            self.jobs[job_id] = {
                "status": "queued",
                "progress": 0.0,
                "npz": [],
                "bvh": [],
                "error": None,
                "started_at": None,
                "finished_at": None,
            }
        self.queue.put((job_id, req))
        return job_id

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] in ("queued", "running"):
                job["cancel"] = True
                if job["status"] == "queued":
                    job["status"] = "canceled"
        return self.public(job_id)

    def public(self, job_id):
        with self.lock:
            return {k: v for k, v in self.jobs[job_id].items() if k != "cancel"}

    def worker(self):
        while True:
            job_id, req = self.queue.get()
            job = self.jobs[job_id]
            if job["status"] == "canceled":
                continue
            while self.status == "loading":
                time.sleep(0.2)
            try:
                if self.status != "ready":
                    raise RuntimeError(f"service failed to load: {self.error}")
                job.update(status="running", started_at=time.time())
                npz, bvh = self.run(req, job)
                job.update(status="done", progress=1.0, npz=npz, bvh=bvh)
            except Canceled:
                job["status"] = "canceled"
            except Exception as e:
                traceback.print_exc()
                job.update(status="error", error=f"{type(e).__name__}: {e}")
            job["finished_at"] = time.time()
            self.last_activity = time.monotonic()

    def run(self, req, job):
        """generate exactly like kimodo_gen"""
        import torch
        from kimodo.constraints import load_constraints_lst
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.tools import seed_everything

        model = self.get_model(req["model"])
        constraint_lst = load_constraints_lst(req["constraints"], model.skeleton) if req["constraints"] else []
        seed_everything(req["seed"])

        def progress(indices):
            indices = list(indices)
            for n, i in enumerate(indices):
                if job.get("cancel"):
                    raise Canceled()
                job["progress"] = n / len(indices)
                yield i

        kwargs = {}
        if req["root_margin"] is not None:
            kwargs["root_margin"] = float(req["root_margin"])
        if req["first_heading_angle"] is not None:
            kwargs["first_heading_angle"] = float(req["first_heading_angle"])
        generate_inner = model._generate
        model._generate = lambda *a, **kw: generate_inner(*a, **{"progress_bar": progress, **kw})
        try:
            output = model(
                req["texts"],
                req["num_frames"],
                constraint_lst=constraint_lst,
                num_denoising_steps=req["steps"],
                num_samples=req["num_samples"],
                multi_prompt=True,
                num_transition_frames=NUM_TRANSITION_FRAMES,
                post_processing=req["postprocess"],
                return_numpy=True,
                cfg_type="separated",
                cfg_weight=[req["cfg_text"], req["cfg_constraint"]],
                progress_bar=progress,
                **kwargs,
            )
        finally:
            del model._generate

        os.makedirs(req["output_dir"], exist_ok=True)
        n = int(output["posed_joints"].shape[0])
        npz_paths, bvh_paths = [], []
        for i in range(n):
            single = {
                k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n else v)
                for k, v in output.items()
            }
            path = os.path.join(req["output_dir"], f"sample_{i}.npz")
            save_kimodo_npz(path, single)
            npz_paths.append(path)
            if req["bvh"]:
                bvh_paths.append(self.save_bvh(model, single, path[:-4] + ".bvh"))
        with open(os.path.join(req["output_dir"], "request.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in req.items() if k != "output_dir"}, f)
        torch.cuda.empty_cache()
        return npz_paths, bvh_paths

    @staticmethod
    def save_bvh(model, single, path):
        """bvh with the standard t-pose rest, in cm"""
        import torch
        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots

        skeleton = model.skeleton
        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to("cuda:0")
        rots = torch.from_numpy(single["global_rot_mats"]).to("cuda:0")
        pos = torch.from_numpy(single["posed_joints"]).to("cuda:0")
        save_motion_bvh(
            path,
            global_rots_to_local_rots(rots, skeleton),
            pos[:, skeleton.root_idx, :],
            skeleton=skeleton,
            fps=model.fps,
            standard_tpose=True,
        )
        return path


def make_handler(service, server_ref):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body):
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, service.health())
            parts = self.path.strip("/").split("/")
            if len(parts) == 2 and parts[0] == "jobs":
                if parts[1] not in service.jobs:
                    return self.send(404, {"error": f"no job {parts[1]}"})
                return self.send(200, service.public(parts[1]))
            self.send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as e:
                return self.send(400, {"error": f"invalid JSON: {e}"})
            parts = self.path.strip("/").split("/")
            if self.path == "/generate":
                try:
                    return self.send(200, {"job_id": service.submit(body)})
                except RequestError as e:
                    return self.send(400, {"error": str(e)})
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                if parts[1] not in service.jobs:
                    return self.send(404, {"error": f"no job {parts[1]}"})
                return self.send(200, service.cancel(parts[1]))
            if self.path == "/clients/register":
                if not isinstance(body.get("id"), str) or not isinstance(body.get("pid"), int):
                    return self.send(400, {"error": "register needs a string id and an integer pid"})
                service.register(body["id"], body["pid"])
                return self.send(200, {"clients": len(service.clients)})
            if self.path == "/clients/unregister":
                service.unregister(str(body.get("id")))
                return self.send(200, {"clients": len(service.clients)})
            if self.path == "/shutdown":
                self.send(200, {})
                threading.Thread(target=server_ref[0].shutdown, daemon=True).start()
                return
            self.send(404, {"error": f"unknown path {self.path}"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("KIMODO_SERVICE_PORT") or 44372),
        help="default from env KIMODO_SERVICE_PORT, else 44372",
    )
    ap.add_argument(
        "--text-encoder-device",
        choices=("cuda", "cpu"),
        default=default_text_encoder_device(),
        help="cpu: run the text encoder on the CPU, for GPUs with less than ~16 GB "
        "(default from env KIMODO_TEXT_ENCODER_DEVICE, else settings.json, else cuda)",
    )
    ap.add_argument(
        "--idle-timeout",
        type=float,
        default=float(os.environ.get("KIMODO_SERVICE_IDLE_MINUTES", 30)),
        help="minutes without jobs before the service exits; 0 = never "
        "(default from env KIMODO_SERVICE_IDLE_MINUTES, else 30)",
    )
    args = ap.parse_args()

    set_default_env()
    if args.text_encoder_device == "cpu":
        os.environ["TEXT_ENCODER_DEVICE"] = "cpu"
    service = Service(text_encoder_device=args.text_encoder_device, idle_timeout=args.idle_timeout * 60)
    server_ref = []
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service, server_ref))
    server_ref.append(server)
    threading.Thread(target=service.load, daemon=True).start()
    threading.Thread(target=service.worker, daemon=True).start()
    threading.Thread(target=service.monitor, args=(server.shutdown,), daemon=True).start()
    print(f"Kimodo service listening on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    main()
