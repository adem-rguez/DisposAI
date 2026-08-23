"""
Diffusers text-to-image HTTP server.

Spawned as a subprocess by the Rust `sd-backend` crate. Loads a `diffusers`
Stable Diffusion pipeline once at startup (from a single `.safetensors` file
or an HF-repo directory) and serves generation requests over a stdlib HTTP
server.

Usage:
    python diffusers_server.py --model-path <path/to/model.safetensors|dir> --port <port>

Protocol:
    POST /generate
        body: {"prompt": "...", "negative_prompt": "...", "steps": 20,
                "guidance_scale": 7.5, "width": 512, "height": 512, "seed": null}
        200:  {"image_base64": "<base64 png>"}
        4xx/5xx: {"error": "..."}

Prints "READY" to stdout once the pipeline is loaded and the server is
listening. The Rust backend watches for this line.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time

# Import torch up front: creating a CUDA session with a different DLL-loading
# order than torch expects can fail with WinError 127. Loading torch first
# and injecting the nvidia package DLL directories wins. Mirrors
# kokoro_tts_server.py / hf_transformers_server.py's import ordering.
try:
    import torch  # noqa: F401
except ImportError:
    pass

if sys.platform == "win32":
    import importlib.util as _ilu
    for _mod in ("nvidia.cu13", "nvidia.cudnn"):
        _spec = _ilu.find_spec(_mod)
        if _spec and _spec.submodule_search_locations:
            for _loc in _spec.submodule_search_locations:
                for _bin in ("bin", os.path.join("bin", "x86_64")):
                    _p = os.path.join(_loc, _bin)
                    if os.path.isdir(_p):
                        os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")
                        os.add_dll_directory(_p)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionPipeline
except ImportError as e:
    print(f"Failed to import diffusers/torch: {e}", file=sys.stderr)
    sys.exit(1)

from PIL import Image

PIPE = None
IMG2IMG_PIPE = None
DEVICE = None

PROGRESS_LOCK = threading.Lock()
PROGRESS = {"status": "idle"}


def _progress_reset(total: int = 0):
    with PROGRESS_LOCK:
        PROGRESS.clear()
        PROGRESS.update({
            "job_id": "",
            "modality": "image",
            "phase": "queued",
            "step": 0,
            "total": total,
            "percent": -1 if total <= 0 else 0.0,
            "status": "running",
            "updated_at": int(time.time() * 1000),
        })


def _progress_update(**kwargs):
    with PROGRESS_LOCK:
        PROGRESS.update(kwargs)
        total = PROGRESS.get("total", 0)
        step = PROGRESS.get("step", 0)
        PROGRESS["percent"] = (step / total * 100.0) if total > 0 else -1
        PROGRESS["updated_at"] = int(time.time() * 1000)


def _progress_read() -> dict:
    with PROGRESS_LOCK:
        return dict(PROGRESS)


def generate(req: dict) -> "PIL.Image.Image":  # noqa: F821
    prompt = req.get("prompt", "")
    negative_prompt = req.get("negative_prompt") or None
    steps = int(req.get("steps") or 20)
    guidance_scale = float(req.get("guidance_scale") or 7.5)
    width = int(req.get("width") or 512)
    height = int(req.get("height") or 512)
    seed = req.get("seed")
    init_image_b64 = req.get("init_image_base64")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    _progress_update(phase="generating", status="running", step=0, total=steps)

    def _cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        _progress_update(step=step_index + 1, total=steps)
        return callback_kwargs

    def _cb_legacy(step, timestep, latents):
        _progress_update(step=step + 1, total=steps)

    if init_image_b64:
        global IMG2IMG_PIPE
        if IMG2IMG_PIPE is None:
            IMG2IMG_PIPE = AutoPipelineForImage2Image.from_pipe(PIPE)
        init_image = Image.open(io.BytesIO(base64.b64decode(init_image_b64))).convert("RGB")
        strength = float(req.get("strength") or 0.75)
        try:
            result = IMG2IMG_PIPE(
                prompt=prompt,
                image=init_image,
                strength=strength,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_cb_on_step_end,
            )
        except TypeError:
            result = IMG2IMG_PIPE(
                prompt=prompt,
                image=init_image,
                strength=strength,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=_cb_legacy,
                callback_steps=1,
            )
        return result.images[0]

    try:
        result = PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=_cb_on_step_end,
        )
    except TypeError:
        result = PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback=_cb_legacy,
            callback_steps=1,
        )
    return result.images[0]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/progress":
            self._send_json(200, _progress_read())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))

            if not req.get("prompt"):
                self._send_json(400, {"error": "'prompt' is required"})
                return

            _progress_reset()

            image = generate(req)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

            _progress_update(status="done", phase="done", step=PROGRESS.get("total", 0))

            self._send_json(200, {"image_base64": image_base64})
        except Exception as e:
            _progress_update(status="error", phase="error", message=str(e))
            self._send_json(500, {"error": str(e)})


def load_pipeline(model_path: str):
    if os.path.isdir(model_path):
        return AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float16)
    return StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16)


def main():
    global PIPE, DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading diffusers pipeline from {args.model_path}...", file=sys.stderr)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIPE = load_pipeline(args.model_path)
    PIPE = PIPE.to(DEVICE)

    print(f"diffusers pipeline loaded on device: {DEVICE}", file=sys.stderr)

    # Refuse to share a port with an orphaned server from an earlier daemon run —
    # on Windows SO_REUSEADDR would let the bind succeed and split requests between
    # the two processes, so requests silently land on the stale one.
    class ExclusiveHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = False

    server = ExclusiveHTTPServer(("127.0.0.1", args.port), Handler)
    print("READY", flush=True)
    sys.stdout.flush()

    server.serve_forever()


if __name__ == "__main__":
    main()
