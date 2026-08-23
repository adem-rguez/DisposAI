"""
Diffusers text/image-to-video HTTP server.

Spawned as a subprocess by the Rust `video-backend` crate. Loads a `diffusers`
`DiffusionPipeline` once at startup (from an HF-repo directory, or a single
model file `diffusers` knows how to load) and serves generation requests over
a stdlib HTTP server.

This is intentionally ONE generic code path with no per-architecture
branching: `diffusers.DiffusionPipeline.from_pretrained()` auto-detects the
right pipeline class from the repo's `model_index.json`, and the actual
`__call__` kwargs are filtered through `inspect.signature()` at request time,
so the same script serves Wan2.1, CogVideoX, LTX-Video, Mochi-1,
HunyuanVideo, AnimateDiff, Stable Video Diffusion, and future video
architectures diffusers adds, without code changes here.

Usage:
    python video_diffusers_server.py --model-path <path/to/model_dir> --port <port>

Protocol:
    POST /generate
        body: {"prompt": "...", "negative_prompt": "...", "image_b64": "...",
                "num_frames": 16, "height": 512, "width": 512,
                "num_inference_steps": 25, "guidance_scale": 7.5, "fps": 8,
                "seed": null, "extra": {...}}
        200:  {"video_b64": "<base64 mp4>", "fps": 8, "num_frames": 16}
        4xx/5xx: {"error": "..."}
    GET /schema -> {"params": [...]} describing the fields the *loaded*
        pipeline's `__call__` actually accepts (introspected, not a static
        list — see `_schema_params_for_loaded_pipe`).
    GET /progress -> coarse job status (see PROGRESS below).
    GET /health -> {"status": "ok"}

Prints "READY" to stdout once the pipeline is loaded and the server is
listening. The Rust backend watches for this line.
"""

import argparse
import base64
import inspect
import io
import json
import os
import sys
import tempfile
import threading
import time

# Import torch up front: creating a CUDA session with a different DLL-loading
# order than torch expects can fail with WinError 127. Loading torch first
# and injecting the nvidia package DLL directories wins. Mirrors
# diffusers_server.py / kokoro_tts_server.py / hf_transformers_server.py's
# import ordering.
try:
    import torch  # noqa: F401
except ImportError:
    pass

if sys.platform == "win32":
    import importlib.util as _ilu
    for _mod in ("nvidia.cu13", "nvidia.cudnn"):
        try:
            _spec = _ilu.find_spec(_mod)
        except (ImportError, ValueError):
            _spec = None
        if _spec and _spec.submodule_search_locations:
            for _loc in _spec.submodule_search_locations:
                for _bin in ("bin", os.path.join("bin", "x86_64")):
                    _p = os.path.join(_loc, _bin)
                    if os.path.isdir(_p):
                        os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")
                        os.add_dll_directory(_p)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import diffusers
    from diffusers.utils import export_to_video
except ImportError as e:
    print(f"Failed to import diffusers/torch: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError as e:
    print(f"Failed to import PIL: {e}", file=sys.stderr)
    Image = None

PIPE = None
DEVICE = None

# Common fields every video pipeline may support, filtered through the loaded
# pipeline's actual `__call__` signature before use (both here, for `/schema`,
# and in `_build_call_kwargs`, for the actual generation call) so the fields
# advertised to the UI always match what the loaded architecture accepts.
PARAM_SCHEMA = [
    {"name": "prompt", "type": "str", "required": False, "default": None, "description": "Text prompt (text-to-video / most img2video pipelines)."},
    {"name": "negative_prompt", "type": "str", "required": False, "default": "", "description": "Negative text prompt, if the pipeline supports classifier-free guidance."},
    {"name": "image", "type": "image_b64", "required": False, "default": None, "description": "Conditioning image for img2video pipelines (e.g. Stable Video Diffusion)."},
    {"name": "num_frames", "type": "int", "required": False, "default": 16, "description": "Number of frames to generate."},
    {"name": "height", "type": "int", "required": False, "default": None, "description": "Output frame height in pixels."},
    {"name": "width", "type": "int", "required": False, "default": None, "description": "Output frame width in pixels."},
    {"name": "num_inference_steps", "type": "int", "required": False, "default": 25, "description": "Denoising steps."},
    {"name": "guidance_scale", "type": "float", "required": False, "default": 9.0, "description": "Classifier-free guidance scale."},
    {"name": "fps", "type": "int", "required": False, "default": 8, "description": "Frames-per-second used to encode the output MP4 (default 8)."},
    {"name": "seed", "type": "int", "required": False, "default": -1, "description": "RNG seed."},
]

# Params whose runtime handling isn't a literal `PIPE.__call__` kwarg name:
# `fps` only controls MP4 export (never passed to the pipeline) so it's
# always offered, and `seed` is only meaningful when the pipeline accepts a
# `generator` kwarg.
_SCHEMA_ALWAYS_INCLUDED = {"fps"}
_SCHEMA_NAME_ALIAS = {"seed": "generator"}


def _schema_params_for_loaded_pipe() -> list:
    """Filter PARAM_SCHEMA down to the fields the loaded pipeline's
    `__call__` signature actually accepts, via the same `inspect.signature`
    introspection `_build_call_kwargs` uses at request time. No
    per-architecture branching: whatever the pipeline's signature reports is
    what gets advertised."""
    if PIPE is None:
        return PARAM_SCHEMA
    accepted = inspect.signature(PIPE.__call__).parameters
    result = []
    for p in PARAM_SCHEMA:
        name = p["name"]
        if name in _SCHEMA_ALWAYS_INCLUDED:
            result.append(p)
            continue
        if _SCHEMA_NAME_ALIAS.get(name, name) in accepted:
            result.append(p)
    return result

PROGRESS_LOCK = threading.Lock()
PROGRESS = {"status": "idle"}


def _progress_reset(total: int = 0):
    with PROGRESS_LOCK:
        PROGRESS.clear()
        PROGRESS.update({
            "job_id": "",
            "modality": "video",
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


def _build_call_kwargs(req: dict) -> dict:
    """Filter the request body's fields (plus `extra`) through the loaded
    pipeline's actual `__call__` signature, so only kwargs the specific
    architecture accepts are ever passed. This is what lets one script serve
    text-to-video pipelines (which take `prompt` but not `image`), img2video
    pipelines like SVD (which take `image` but not `prompt`), and
    architectures whose frame-count kwarg isn't literally `num_frames`,
    without any per-architecture branching."""
    accepted = inspect.signature(PIPE.__call__).parameters

    candidate = {
        "prompt": req.get("prompt") or None,
        "negative_prompt": req.get("negative_prompt") or None,
        "num_frames": req.get("num_frames"),
        "height": req.get("height"),
        "width": req.get("width"),
        "num_inference_steps": req.get("num_inference_steps"),
        "guidance_scale": req.get("guidance_scale"),
    }

    image_b64 = req.get("image_b64")
    if image_b64 and Image is not None:
        img_bytes = base64.b64decode(image_b64)
        candidate["image"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    seed = req.get("seed")
    if seed is not None and "generator" in accepted:
        candidate["generator"] = torch.Generator(device=DEVICE).manual_seed(int(seed))

    kwargs = {k: v for k, v in candidate.items() if v is not None and k in accepted}

    # Merge in adapter-specific passthrough params last, also filtered
    # through the signature, so users can target model-specific kwargs
    # (e.g. CogVideoX's `use_dynamic_cfg`) without code changes here.
    extra = req.get("extra") or {}
    for k, v in extra.items():
        if k in accepted and v is not None:
            kwargs[k] = v

    return kwargs


def generate(req: dict):
    fps = int(req.get("fps") or 8)

    def _cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        _progress_update(step=step_index + 1)
        return callback_kwargs

    def _cb_legacy(step, timestep, latents):
        _progress_update(step=step + 1)

    kwargs = _build_call_kwargs(req)
    accepted = inspect.signature(PIPE.__call__).parameters

    steps = kwargs.get("num_inference_steps") or 25
    _progress_update(phase="generating", status="running", step=0, total=steps)

    if "callback_on_step_end" in accepted:
        kwargs["callback_on_step_end"] = _cb_on_step_end
    elif "callback" in accepted:
        kwargs["callback"] = _cb_legacy
        if "callback_steps" in accepted:
            kwargs["callback_steps"] = 1

    result = PIPE(**kwargs)
    frames = result.frames[0]
    return frames, fps


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
        elif self.path == "/schema":
            self._send_json(200, {"params": _schema_params_for_loaded_pipe()})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            req = json.loads(raw.decode("utf-8")) if raw else {}

            if not req.get("prompt") and not req.get("image_b64"):
                self._send_json(400, {"error": "either 'prompt' or 'image_b64' is required"})
                return

            _progress_reset()

            frames, fps = generate(req)

            tmp_path = os.path.join(tempfile.gettempdir(), f"dispos_video_{os.getpid()}_{int(time.time() * 1000)}.mp4")
            try:
                export_to_video(frames, tmp_path, fps=fps)
                with open(tmp_path, "rb") as f:
                    video_bytes = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            video_b64 = base64.b64encode(video_bytes).decode("ascii")

            _progress_update(status="done", phase="done", step=PROGRESS.get("total", 0))

            self._send_json(200, {"video_b64": video_b64, "fps": fps, "num_frames": len(frames)})
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            _progress_update(status="error", phase="error", message=str(e))
            self._send_json(500, {"error": str(e)})


def load_pipeline(model_path: str):
    pipe = diffusers.DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

    # Opportunistically apply whatever VRAM-saving hooks the loaded pipeline
    # exposes. Never hardcode which architectures have them — just probe.
    # IMPORTANT: don't call `.to("cuda")` unconditionally first — that would
    # move the entire fp16 pipeline onto the GPU before any offload kicks in,
    # which can OOM an 8GB card on large video models. `enable_model_cpu_offload`
    # / `enable_sequential_cpu_offload` manage device placement themselves
    # (moving modules to GPU only as needed during inference), so `.to(DEVICE)`
    # must only be used as a fallback when neither offload hook is available.
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pipe = pipe.to(DEVICE)

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_slicing"):
            try:
                vae.enable_slicing()
            except Exception:
                pass
        if hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception:
                pass

    return pipe


def main():
    global PIPE, DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading diffusers video pipeline from {args.model_path}...", file=sys.stderr)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIPE = load_pipeline(args.model_path)

    print(f"diffusers video pipeline loaded on device: {DEVICE}", file=sys.stderr)

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
