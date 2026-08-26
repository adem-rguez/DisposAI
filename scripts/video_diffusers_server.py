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
import re
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


# Base repo used only as a source for the shared non-quantized LTX-Video
# components (text_encoder/tokenizer/scheduler, and vae if none is found
# locally) when loading a bare .gguf/.safetensors LTX transformer that isn't
# packaged as a full diffusers pipeline directory.
LTX_BASE_REPO = "Lightricks/LTX-Video"

# Weights file extensions considered "main weights" candidates when scanning
# a directory for a loose (non model_index.json) checkpoint.
_WEIGHTS_EXTS = {".gguf", ".safetensors", ".ckpt", ".bin", ".pt", ".pth"}
_VAE_EXTS = {".gguf", ".safetensors", ".bin", ".pt", ".pth"}

# Set from --ltx-components-dir. When it holds a complete set of the files
# below, the LTX-Video text_encoder/tokenizer/scheduler are built from these
# local files instead of silently auto-downloading them from LTX_BASE_REPO
# (see crates/backends/video-backend's `detect_video_components`, which lets
# the UI offer these as a visible, user-triggered download with progress).
LTX_COMPONENTS_DIR = None

_LTX_TEXT_ENCODER_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
)
_LTX_TOKENIZER_FILES = ("added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json")
_LTX_SCHEDULER_FILES = ("scheduler_config.json",)


def _ltx_local_components_complete(components_dir):
    if not components_dir:
        return False
    checks = (
        ("text_encoder", _LTX_TEXT_ENCODER_FILES),
        ("tokenizer", _LTX_TOKENIZER_FILES),
        ("scheduler", _LTX_SCHEDULER_FILES),
    )
    return all(
        os.path.isfile(os.path.join(components_dir, subdir, name))
        for subdir, files in checks
        for name in files
    )


def _load_ltx_local_pipeline_components(components_dir):
    """Build text_encoder/tokenizer/scheduler from a complete local
    components dir (see `_ltx_local_components_complete`)."""
    from transformers import T5EncoderModel, T5TokenizerFast

    text_encoder = T5EncoderModel.from_pretrained(
        os.path.join(components_dir, "text_encoder"), torch_dtype=torch.float16,
    )
    tokenizer = T5TokenizerFast.from_pretrained(os.path.join(components_dir, "tokenizer"))
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(components_dir, "scheduler"))
    return {"text_encoder": text_encoder, "tokenizer": tokenizer, "scheduler": scheduler}


def _filename_is_vae(name: str) -> bool:
    """True if `name` (a bare filename) identifies a VAE weights file: "vae"
    appears as a distinct word-boundary token in the filename stem (split on
    `_`/`-`/`.`), e.g. "vae.safetensors" or "ltx_vae.gguf". Case-insensitive.
    Mirrors `filename_is_vae` in crates/daemon-core/src/http.rs."""
    stem = os.path.splitext(name)[0].lower()
    return "vae" in re.split(r"[_\-.]", stem)


def _find_loose_main_weights(directory: str):
    """Find a loose (non model_index.json) main weights file directly inside
    `directory`, skipping VAE and mmproj companion files."""
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    for entry in sorted(entries):
        full = os.path.join(directory, entry)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext not in _WEIGHTS_EXTS:
            continue
        if _filename_is_vae(entry) or "mmproj" in entry.lower():
            continue
        return full
    return None


def _find_sibling_vae(main_weights_path: str):
    """Look for a same-directory sibling VAE file for `main_weights_path`
    (e.g. LTX-Video repos ship `ltx-video.gguf` + `vae.gguf` as loose
    root-level siblings). Mirrors `find_sibling_vae` in http.rs."""
    directory = os.path.dirname(main_weights_path) or "."
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    for entry in sorted(entries):
        full = os.path.join(directory, entry)
        if full == main_weights_path or not os.path.isfile(full):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in _VAE_EXTS and _filename_is_vae(entry):
            return full
    return None


def _load_ltx_pipeline_from_single_files(transformer_path: str):
    """Build an LTXPipeline from a loose (bare) transformer weights file
    (`.gguf` or `.safetensors`), with an optional sibling VAE file, pulling
    any other required components (text_encoder/tokenizer/scheduler, and vae
    if not found locally) from LTX_BASE_REPO."""
    required_symbols = ("LTXVideoTransformer3DModel", "AutoencoderKLLTXVideo", "GGUFQuantizationConfig", "LTXPipeline")
    missing = [s for s in required_symbols if not hasattr(diffusers, s)]
    if missing:
        raise RuntimeError(
            f"Cannot load LTX-Video model from loose weights file {transformer_path!r}: "
            f"installed diffusers ({getattr(diffusers, '__version__', '?')}) is missing "
            f"{', '.join('diffusers.' + s for s in missing)}. Upgrade diffusers to a version "
            "with LTX-Video single-file loading support."
        )

    ext = os.path.splitext(transformer_path)[1].lower()
    if ext not in (".gguf", ".safetensors"):
        raise RuntimeError(
            f"Cannot load LTX-Video transformer from {transformer_path!r}: unsupported "
            f"extension {ext!r} (expected .gguf or .safetensors)."
        )

    try:
        if ext == ".gguf":
            transformer = diffusers.LTXVideoTransformer3DModel.from_single_file(
                transformer_path,
                quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=torch.float16),
                torch_dtype=torch.float16,
            )
        else:
            transformer = diffusers.LTXVideoTransformer3DModel.from_single_file(
                transformer_path, torch_dtype=torch.float16,
            )

        vae_path = _find_sibling_vae(transformer_path)
        vae = None
        if vae_path:
            # diffusers' LTX VAE conversion expects checkpoint keys prefixed with
            # "vae." (it filters a combined pipeline checkpoint by that prefix),
            # but standalone VAE-only files like this one store keys unprefixed
            # (e.g. "encoder.conv_out.conv.weight"). from_single_file accepts an
            # in-memory state dict directly, so prefix it ourselves rather than
            # writing a rewritten copy of a multi-GB file to disk.
            from safetensors.torch import load_file as _load_safetensors_file

            vae_state_dict = _load_safetensors_file(vae_path)
            if not any(k.startswith("vae.") for k in vae_state_dict):
                vae_state_dict = {f"vae.{k}": v for k, v in vae_state_dict.items()}

            # Do NOT pass `config=` here: diffusers auto-detects the LTX-Video
            # *version* (0.9.0 / 0.9.1 / 0.9.5 / ...) from marker keys like
            # "vae.decoder.last_scale_shift_table", each version having a
            # differently-shaped VAE. Forcing a fixed base-repo config caused a
            # 0.9.1 VAE checkpoint to be loaded against the 0.9.0 architecture
            # (mismatched channel counts). Auto-detection only works now that
            # the checkpoint keys carry the "vae." prefix it looks for.
            vae = diffusers.AutoencoderKLLTXVideo.from_single_file(
                vae_state_dict, torch_dtype=torch.float16,
            )

        pipeline_kwargs = {"transformer": transformer, "torch_dtype": torch.float16}
        if vae is not None:
            pipeline_kwargs["vae"] = vae

        if _ltx_local_components_complete(LTX_COMPONENTS_DIR):
            print(f"Loading LTX-Video pipeline components from local cache: {LTX_COMPONENTS_DIR}", file=sys.stderr)
            pipeline_kwargs.update(_load_ltx_local_pipeline_components(LTX_COMPONENTS_DIR))
            pipe = diffusers.LTXPipeline(**pipeline_kwargs)
        else:
            pipe = diffusers.LTXPipeline.from_pretrained(LTX_BASE_REPO, **pipeline_kwargs)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to load LTX-Video pipeline from local weights {transformer_path!r} "
            f"(base repo {LTX_BASE_REPO!r} used for missing components): {e}"
        ) from e

    return pipe


def load_pipeline(model_path: str):
    is_full_pipeline_dir = os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "model_index.json"))

    loose_transformer_path = None
    if not is_full_pipeline_dir:
        if os.path.isfile(model_path) and os.path.splitext(model_path)[1].lower() in (".gguf", ".safetensors"):
            loose_transformer_path = model_path
        elif os.path.isdir(model_path):
            loose_transformer_path = _find_loose_main_weights(model_path)

    if loose_transformer_path is not None:
        pipe = _load_ltx_pipeline_from_single_files(loose_transformer_path)
    else:
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
    global PIPE, DEVICE, LTX_COMPONENTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ltx-components-dir", default=None)
    args = parser.parse_args()

    LTX_COMPONENTS_DIR = args.ltx_components_dir

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
