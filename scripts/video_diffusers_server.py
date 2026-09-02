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

# The pipeline class, resolved from the checkpoint's header before any weights
# are read (see `_resolve_pipeline_class`). `/schema` answers from this while
# PIPE is still loading, so the port can be bound and READY signalled up front.
PIPE_CLASS = None
# Set once the background load settles: PIPE is usable, or PIPE_LOAD_ERROR
# explains why it isn't. /generate waits on this rather than racing the load.
PIPE_READY = threading.Event()
PIPE_LOAD_ERROR = None

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
    """Filter PARAM_SCHEMA down to the fields the pipeline's `__call__`
    signature actually accepts, via the same `inspect.signature` introspection
    `_build_call_kwargs` uses at request time. No per-architecture branching:
    whatever the signature reports is what gets advertised.

    Falls back to the pipeline *class* while the weights are still loading.
    The signature is a property of the class, not of the instance, so this
    answers `/schema` accurately from the moment the server binds — which is
    what lets the studio render the parameter fields at model-select time
    instead of only once a generation has forced the load to finish."""
    pipe_or_class = PIPE if PIPE is not None else PIPE_CLASS
    if pipe_or_class is None:
        return PARAM_SCHEMA
    accepted = inspect.signature(pipe_or_class.__call__).parameters
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

# Set by POST /cancel, checked from the diffusers step callback so an
# in-flight /generate can be aborted early (the HTTP server is threaded, so
# /cancel can be handled on its own thread while /generate is still running).
CANCEL_EVENT = threading.Event()


class GenerationCancelled(Exception):
    pass


def _progress_reset(total: int = 0):
    CANCEL_EVENT.clear()
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
    # The weights load on a background thread so `/schema` stays answerable
    # immediately (see `main`), so a generation that arrives first has to wait
    # for it — and report the load's own error rather than an opaque failure
    # against a PIPE that is still None.
    if not PIPE_READY.is_set():
        _progress_update(phase="loading", status="running")
        PIPE_READY.wait()
    if PIPE_LOAD_ERROR is not None:
        raise RuntimeError(PIPE_LOAD_ERROR)

    fps = int(req.get("fps") or 8)

    def _cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        if CANCEL_EVENT.is_set():
            raise GenerationCancelled()
        _progress_update(step=step_index + 1)
        return callback_kwargs

    def _cb_legacy(step, timestep, latents):
        if CANCEL_EVENT.is_set():
            raise GenerationCancelled()
        _progress_update(step=step + 1)

    kwargs = _build_call_kwargs(req)
    accepted = inspect.signature(PIPE.__call__).parameters
    _precompute_prompt_embeds(kwargs, accepted)

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
        if self.path == "/cancel":
            CANCEL_EVENT.set()
            self._send_json(200, {"status": "cancelling"})
            return
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
        except GenerationCancelled:
            _progress_update(status="cancelled", phase="cancelled")
            self._send_json(499, {"error": "Generation cancelled", "cancelled": True})
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

# Base repo used the same way as LTX_BASE_REPO above, but for Wan2.2 TI2V-5B
# loose weights files. No local components-dir caching exists for Wan (that's
# an LTX-only feature) — this is the fallback source for whatever components
# aren't overridden. Only the 5B variant; `_WAN_VARIANTS` below covers the rest.
WAN_BASE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Wan comes in several architectures that share a checkpoint layout but differ
# in width and in which transformer/pipeline classes can hold them. Picking one
# repo for all of them builds the wrong module and fails with a shape mismatch
# (a 1.3B checkpoint against the 5B config reports "expected 3072, got 1536").
# Keyed by the model type ids diffusers' own `infer_diffusers_model_type` uses,
# with repos taken from its `DIFFUSERS_DEFAULT_PIPELINE_PATHS`.
_WAN_VARIANTS = {
    "wan-t2v-1.3B": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "WanTransformer3DModel", "WanPipeline"),
    "wan-t2v-14B": ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "WanTransformer3DModel", "WanPipeline"),
    "wan-i2v-14B": ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "WanTransformer3DModel", "WanImageToVideoPipeline"),
    "wan-vace-1.3B": ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "WanVACETransformer3DModel", "WanVACEPipeline"),
    "wan-vace-14B": ("Wan-AI/Wan2.1-VACE-14B-diffusers", "WanVACETransformer3DModel", "WanVACEPipeline"),
    "wan-ti2v-5B": (WAN_BASE_REPO, "WanTransformer3DModel", "WanPipeline"),
}

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


def _find_text_encoder_gguf(text_encoder_dir: str):
    """Look for a single loose quantized `.gguf` text-encoder file directly
    inside `text_encoder_dir`. Any quant level is accepted; the filename
    itself doesn't need to identify it as a text encoder since it already
    lives in a `text_encoder` subdir."""
    try:
        entries = sorted(os.listdir(text_encoder_dir))
    except OSError:
        return None
    for entry in entries:
        if entry.lower().endswith(".gguf"):
            full = os.path.join(text_encoder_dir, entry)
            if os.path.isfile(full):
                return full
    return None


def _ltx_local_components_complete(components_dir):
    if not components_dir:
        return False
    text_encoder_dir = os.path.join(components_dir, "text_encoder")
    text_encoder_ok = _find_text_encoder_gguf(text_encoder_dir) is not None or all(
        os.path.isfile(os.path.join(text_encoder_dir, name)) for name in _LTX_TEXT_ENCODER_FILES
    )
    checks = (
        ("tokenizer", _LTX_TOKENIZER_FILES),
        ("scheduler", _LTX_SCHEDULER_FILES),
    )
    return text_encoder_ok and all(
        os.path.isfile(os.path.join(components_dir, subdir, name))
        for subdir, files in checks
        for name in files
    )


def _load_ltx_local_pipeline_components(components_dir):
    """Build text_encoder/tokenizer/scheduler from a complete local
    components dir (see `_ltx_local_components_complete`)."""
    from transformers import T5EncoderModel, T5TokenizerFast

    text_encoder_dir = os.path.join(components_dir, "text_encoder")
    text_encoder_gguf = _find_text_encoder_gguf(text_encoder_dir)
    if text_encoder_gguf is not None:
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_dir, gguf_file=os.path.basename(text_encoder_gguf), torch_dtype=torch.float16,
        )
    else:
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_dir, torch_dtype=torch.float16,
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


def _filename_is_text_encoder(name: str) -> bool:
    """True if `name` (a bare filename) identifies a T5/UMT5 text-encoder
    weights file: "t5", "umt5", "text_encoder", or "text-encoder" appears as
    a distinct word-boundary token in the filename stem (split on
    `_`/`-`/`.`). Case-insensitive. Mirrors `_filename_is_vae` above."""
    stem = os.path.splitext(name)[0].lower()
    tokens = set(re.split(r"[_\-.]", stem))
    return "t5" in tokens or "umt5" in tokens or ("encoder" in tokens and "text" in tokens)


def _find_loose_main_weights(directory: str):
    """Find a loose (non model_index.json) main weights file directly inside
    `directory`, skipping VAE, text-encoder, and mmproj companion files."""
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
        if _filename_is_vae(entry) or _filename_is_text_encoder(entry) or "mmproj" in entry.lower():
            continue
        return full
    return None


def _iter_sibling_files(directory: str):
    """Yield (filename, full_path) for files directly in `directory`, then for
    files one level down in its subdirectories. Some GGUF repos park companion
    weights in a subfolder instead of as flat siblings (e.g. QuantStack's
    Wan2.2-TI2V-5B-GGUF ships `VAE/Wan2.2_VAE.safetensors` alongside the
    root-level transformer). Root files are yielded first so flat layouts keep
    their existing match order."""
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return
    for entry in entries:
        full = os.path.join(directory, entry)
        if os.path.isfile(full):
            yield entry, full
    for entry in entries:
        sub = os.path.join(directory, entry)
        if not os.path.isdir(sub):
            continue
        try:
            sub_entries = sorted(os.listdir(sub))
        except OSError:
            continue
        for name in sub_entries:
            full = os.path.join(sub, name)
            if os.path.isfile(full):
                yield name, full


def _find_sibling_vae(main_weights_path: str):
    """Look for a sibling VAE file for `main_weights_path` (e.g. LTX-Video
    repos ship `ltx-video.gguf` + `vae.gguf` as loose root-level siblings),
    searching the same directory and its immediate subdirectories. Mirrors
    `find_sibling_vae` in http.rs."""
    directory = os.path.dirname(main_weights_path) or "."
    for entry, full in _iter_sibling_files(directory):
        if full == main_weights_path:
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in _VAE_EXTS and _filename_is_vae(entry):
            return full
    return None


def _find_sibling_text_encoder(main_weights_path: str):
    """Look for a sibling quantized `.gguf` text-encoder file for
    `main_weights_path` (e.g. some LTX-Video GGUF repos ship
    `ltx-video-q4.gguf` + `t5xxl_fp16-q4_0.gguf` as loose root-level
    siblings), searching the same directory and its immediate
    subdirectories. Any quant level is accepted. Mirrors `_find_sibling_vae`."""
    directory = os.path.dirname(main_weights_path) or "."
    for entry, full in _iter_sibling_files(directory):
        if full == main_weights_path:
            continue
        if entry.lower().endswith(".gguf") and _filename_is_text_encoder(entry):
            return full
    return None


# Matches google/t5-v1_1-xxl (LTX-Video's text encoder architecture), used
# as a fallback config when a manually-overridden text-encoder file has no
# sibling config.json to load a real config from.
_T5_XXL_V1_1_CONFIG = {
    "d_model": 4096, "d_kv": 64, "d_ff": 10240,
    "num_layers": 24, "num_decoder_layers": 24, "num_heads": 64,
    "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
    "dropout_rate": 0.1, "layer_norm_epsilon": 1e-06, "initializer_factor": 1.0,
    "feed_forward_proj": "gated-gelu", "is_encoder_decoder": True,
    "vocab_size": 32128, "tie_word_embeddings": False,
}

# Matches google/umt5-xxl (Wan2.2's text encoder architecture), used as a
# fallback config when a manually-overridden UMT5 text-encoder file has no
# sibling config.json to load a real config from. Mirrors
# _T5_XXL_V1_1_CONFIG above, with UMT5's larger multilingual vocab.
_UMT5_XXL_CONFIG = {
    "d_model": 4096, "d_kv": 64, "d_ff": 10240,
    "num_layers": 24, "num_decoder_layers": 24, "num_heads": 64,
    "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
    "dropout_rate": 0.1, "layer_norm_epsilon": 1e-06, "initializer_factor": 1.0,
    "feed_forward_proj": "gated-gelu", "is_encoder_decoder": True,
    "vocab_size": 256384, "tie_word_embeddings": False,
}


def _load_state_dict_from_file(path: str):
    """Load a checkpoint file (`.gguf`/`.safetensors`/`.ckpt`/`.pt`/`.bin`)
    into a plain state dict, tolerant of the various save layouts.
    `.safetensors` is read via `safetensors.torch.load_file`. `.gguf` is a
    completely different binary container (NOT parseable by the safetensors
    loader) and is read via diffusers' own internal GGUF checkpoint parser,
    the same one `from_single_file(..., quantization_config=...)` uses under
    the hood elsewhere in this file, giving back a plain dict of tensors.
    The rest go through `torch.load`, unwrapping a nested `"state_dict"` key
    if present (some ComfyUI-style checkpoints store more than tensors)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file as _load_safetensors_file

        state_dict = _load_safetensors_file(path)
    elif ext == ".gguf":
        try:
            from diffusers.models.model_loading_utils import load_gguf_checkpoint
        except ImportError:
            raise RuntimeError(
                f"Cannot load GGUF checkpoint {path!r}: installed diffusers "
                f"({getattr(diffusers, '__version__', '?')}) has no "
                "diffusers.models.model_loading_utils.load_gguf_checkpoint. "
                "Upgrade diffusers to a version with GGUF single-file loading support."
            )
        state_dict = load_gguf_checkpoint(path)
    else:
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
            state_dict = state_dict["state_dict"]
    return state_dict


def _load_t5_text_encoder_from_state_dict(path: str):
    """Build a T5EncoderModel from a bare state-dict file (`.safetensors`/
    `.ckpt`/`.pt`/`.bin`) — used for manual text-encoder overrides that
    aren't a `.gguf` (which uses `T5EncoderModel.from_pretrained(...,
    gguf_file=...)` instead). Uses a sibling `config.json` if present,
    otherwise falls back to the known T5-XXL v1.1 config LTX-Video's text
    encoder is built from."""
    from transformers import T5Config, T5EncoderModel

    config_dir = os.path.dirname(path) or "."
    config_json = os.path.join(config_dir, "config.json")
    if os.path.isfile(config_json):
        config = T5Config.from_pretrained(config_dir)
    else:
        config = T5Config(**_T5_XXL_V1_1_CONFIG)

    # Building T5EncoderModel(config) normally allocates the full model in
    # fp32 (~18GB RAM for T5-XXL's 4.7B params) before the state dict below
    # even gets loaded onto it. Constructing on the "meta" device instead
    # skips that allocation entirely (meta tensors have no backing storage),
    # and load_state_dict(..., assign=True) then hands the loaded tensors
    # straight to the model instead of copying into pre-allocated storage.
    with torch.device("meta"):
        model = T5EncoderModel(config)
    state_dict = _load_state_dict_from_file(path)
    # Convert one tensor at a time, in place, instead of building a second
    # full dict via a comprehension — that would hold both the fp8 and fp16
    # copies of the *entire* state dict in memory simultaneously (~2x file
    # size peak) instead of just one tensor's worth of overlap.
    for key in list(state_dict.keys()):
        state_dict[key] = state_dict[key].to(torch.float16)
    load_result = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    print(
        f"Loaded T5 text encoder override from {path!r}: "
        f"{len(load_result.missing_keys)} missing keys, {len(load_result.unexpected_keys)} unexpected keys",
        file=sys.stderr,
    )
    return model


def _load_umt5_text_encoder_from_state_dict(path: str):
    """Build a UMT5EncoderModel from a bare state-dict file (`.safetensors`/
    `.ckpt`/`.pt`/`.bin`) — used for manual Wan2.2 text-encoder overrides
    that aren't a `.gguf`. Mirrors `_load_t5_text_encoder_from_state_dict`
    above, but for UMT5-XXL (Wan's text encoder, not T5)."""
    from transformers import UMT5Config, UMT5EncoderModel

    config_dir = os.path.dirname(path) or "."
    config_json = os.path.join(config_dir, "config.json")
    if os.path.isfile(config_json):
        config = UMT5Config.from_pretrained(config_dir)
    else:
        config = UMT5Config(**_UMT5_XXL_CONFIG)

    with torch.device("meta"):
        model = UMT5EncoderModel(config)
    state_dict = _load_state_dict_from_file(path)
    for key in list(state_dict.keys()):
        state_dict[key] = state_dict[key].to(torch.float16)
    load_result = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    print(
        f"Loaded UMT5 text encoder override from {path!r}: "
        f"{len(load_result.missing_keys)} missing keys, {len(load_result.unexpected_keys)} unexpected keys",
        file=sys.stderr,
    )
    return model


def _detect_gguf_architecture(path: str) -> str:
    """Read the `general.architecture` metadata key out of a `.gguf` file's
    header (without loading any tensor data) to tell apart the different
    video-model families that ship as loose GGUF files, since a bare weights
    file otherwise gives no clue which pipeline it belongs to."""
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(path)
    field = reader.fields.get("general.architecture")
    if field is None:
        raise RuntimeError(
            f"Cannot determine model architecture of GGUF file {path!r}: "
            "no 'general.architecture' metadata key found."
        )
    return field.contents()


def _load_ltx_pipeline_from_single_files(
    transformer_path: str,
    text_encoder_override: str = None,
    vae_override: str = None,
):
    """Build an LTXPipeline from a loose (bare) transformer weights file
    (`.gguf` or `.safetensors`), with optional sibling (or manually
    overridden) VAE and text-encoder files, pulling any other required
    components (text_encoder/tokenizer/scheduler, and vae/text_encoder if
    not found locally) from LTX_BASE_REPO."""
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

        vae_path = vae_override if vae_override else _find_sibling_vae(transformer_path)
        vae = None
        if vae_path:
            # diffusers' LTX VAE conversion expects checkpoint keys prefixed with
            # "vae." (it filters a combined pipeline checkpoint by that prefix),
            # but standalone VAE-only files like this one store keys unprefixed
            # (e.g. "encoder.conv_out.conv.weight"). from_single_file accepts an
            # in-memory state dict directly, so prefix it ourselves rather than
            # writing a rewritten copy of a multi-GB file to disk.
            vae_state_dict = _load_state_dict_from_file(vae_path)
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

        text_encoder_path = text_encoder_override if text_encoder_override else _find_sibling_text_encoder(transformer_path)
        if text_encoder_path:
            if os.path.splitext(text_encoder_path)[1].lower() == ".gguf":
                from transformers import T5EncoderModel

                text_encoder_dir = os.path.dirname(text_encoder_path) or "."
                text_encoder = T5EncoderModel.from_pretrained(
                    text_encoder_dir, gguf_file=os.path.basename(text_encoder_path), torch_dtype=torch.float16,
                )
            else:
                text_encoder = _load_t5_text_encoder_from_state_dict(text_encoder_path)
        else:
            text_encoder = None

        pipeline_kwargs = {"transformer": transformer, "torch_dtype": torch.float16}
        if vae is not None:
            pipeline_kwargs["vae"] = vae
        if text_encoder is not None:
            pipeline_kwargs["text_encoder"] = text_encoder

        if _ltx_local_components_complete(LTX_COMPONENTS_DIR):
            print(f"Loading LTX-Video pipeline components from local cache: {LTX_COMPONENTS_DIR}", file=sys.stderr)
            local_components = _load_ltx_local_pipeline_components(LTX_COMPONENTS_DIR)
            for key, value in local_components.items():
                pipeline_kwargs.setdefault(key, value)
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


# Maps a Wan VAE residual block's inner nn.Sequential index to the diffusers
# WanResidualBlock submodule it holds.
_WAN_VAE_RESIDUAL_SUBMODULES = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}


def _convert_wan22_vae_state_dict(state_dict):
    """Rename a Wan2.2 VAE checkpoint's keys to diffusers' AutoencoderKLWan
    layout. diffusers' own convert_wan_vae_to_diffusers only handles Wan2.1's
    flat blocks — it renames "encoder.downsamples.N" -> "encoder.down_blocks.N"
    and stops there. Wan2.2's residual VAE nests its blocks one level deeper
    ("encoder.downsamples.i.downsamples.j"), so that flat remap leaves the
    inner index in the key and lands tensors in the wrong slots."""
    converted = {}
    for key, value in state_dict.items():
        # Root-level quant convs.
        if key.startswith("conv1."):
            converted["quant_conv." + key.split(".", 1)[1]] = value
            continue
        if key.startswith("conv2."):
            converted["post_quant_conv." + key.split(".", 1)[1]] = value
            continue

        side, rest = key.split(".", 1)
        block = "down_blocks" if side == "encoder" else "up_blocks"
        sampler = "downsampler" if side == "encoder" else "upsampler"
        samples = "downsamples" if side == "encoder" else "upsamples"

        if rest.startswith("conv1."):
            converted[f"{side}.conv_in." + rest.split(".", 1)[1]] = value
            continue
        # head.0 is the output norm, head.2 the output conv.
        if rest.startswith("head."):
            _, idx, tail = rest.split(".", 2)
            converted[f"{side}.norm_out.{tail}" if idx == "0" else f"{side}.conv_out.{tail}"] = value
            continue

        # middle.{0,2} are the two mid-block resnets, middle.1 the attention.
        match = re.match(r"^middle\.(\d+)\.(.+)$", rest)
        if match:
            index, tail = int(match.group(1)), match.group(2)
            residual = re.match(r"^residual\.(\d+)\.(.+)$", tail)
            if residual:
                name = _WAN_VAE_RESIDUAL_SUBMODULES[residual.group(1)]
                converted[f"{side}.mid_block.resnets.{index // 2}.{name}.{residual.group(2)}"] = value
            else:
                converted[f"{side}.mid_block.attentions.0.{tail}"] = value
            continue

        # The nested down/up blocks: inner entries are resnets, except the
        # trailing resample/time_conv one, which is the block's sampler.
        match = re.match(rf"^{samples}\.(\d+)\.{samples}\.(\d+)\.(.+)$", rest)
        if match:
            outer, inner, tail = match.group(1), match.group(2), match.group(3)
            residual = re.match(r"^residual\.(\d+)\.(.+)$", tail)
            if residual:
                name = _WAN_VAE_RESIDUAL_SUBMODULES[residual.group(1)]
                converted[f"{side}.{block}.{outer}.resnets.{inner}.{name}.{residual.group(2)}"] = value
            elif tail.startswith("shortcut."):
                converted[f"{side}.{block}.{outer}.resnets.{inner}.conv_shortcut." + tail.split(".", 1)[1]] = value
            else:
                converted[f"{side}.{block}.{outer}.{sampler}.{tail}"] = value
            continue

        raise KeyError(f"unrecognised Wan VAE checkpoint key {key!r}")
    return converted


def _load_wan_vae_from_state_dict(state_dict):
    """Build an AutoencoderKLWan from a standalone VAE checkpoint's state dict,
    handling both the Wan2.1 (flat) and Wan2.2 (nested residual) layouts."""
    is_wan22 = any(
        ".downsamples.0.downsamples." in k or ".upsamples.0.upsamples." in k
        for k in state_dict
    )
    if not is_wan22:
        return diffusers.AutoencoderKLWan.from_single_file(state_dict, torch_dtype=torch.float16)

    # from_single_file can't convert this layout (see above), so build the
    # model from the base repo's vae/config.json and load the renamed keys
    # directly. Only the small config.json is fetched, not the 2.8 GB weights.
    config = diffusers.AutoencoderKLWan.load_config(WAN_BASE_REPO, subfolder="vae")
    vae = diffusers.AutoencoderKLWan.from_config(config)
    vae.load_state_dict(_convert_wan22_vae_state_dict(state_dict))
    return vae.to(torch.float16)


def _dequant_cache_path(source_path: str) -> str:
    cache_dir = os.path.join(os.path.dirname(source_path) or ".", ".dequant_cache")
    return os.path.join(cache_dir, os.path.basename(source_path) + ".fp16.safetensors")


def _try_load_dequant_cache(source_path: str):
    """Return a cached fp16 state dict for `source_path`'s prior GGUF
    de-quantization, or None if no valid (matching size+mtime) cache exists.
    Never raises — any problem reading the cache just means re-dequantize."""
    cache_path = _dequant_cache_path(source_path)
    if not os.path.isfile(cache_path):
        return None
    try:
        from safetensors import safe_open
        with safe_open(cache_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        src_stat = os.stat(source_path)
        if meta.get("source_size") != str(src_stat.st_size) or meta.get("source_mtime") != str(int(src_stat.st_mtime)):
            return None
        from safetensors.torch import load_file
        return load_file(cache_path)
    except Exception as e:
        print(f"Warning: failed to read de-quant cache {cache_path!r}: {e}", file=sys.stderr)
        return None


def _save_dequant_cache(source_path: str, state_dict: dict):
    """Best-effort: persist a de-quantized fp16 state dict to disk so the
    next load of this exact GGUF file can skip re-de-quantizing it. Never
    raises — a failed cache write must not break model loading."""
    try:
        from safetensors.torch import save_file
        cache_path = _dequant_cache_path(source_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        src_stat = os.stat(source_path)
        metadata = {"source_size": str(src_stat.st_size), "source_mtime": str(int(src_stat.st_mtime))}
        tmp_path = cache_path + ".tmp"
        # safetensors refuses to serialize two keys backed by the same storage,
        # and UMT5 ties `encoder.embed_tokens.weight` to `shared.weight` — which
        # made every write here fail (silently, via the except below) and every
        # restart re-de-quantize from scratch. Keep only the first key for each
        # storage; the load path calls `tie_weights()` to restore the alias.
        seen_storages = set()
        unique = {}
        for key, value in state_dict.items():
            value = value.contiguous()
            storage_id = (value.device.type, value.data_ptr())
            if storage_id in seen_storages:
                continue
            seen_storages.add(storage_id)
            unique[key] = value
        save_file(unique, tmp_path, metadata=metadata)
        os.replace(tmp_path, cache_path)
        print(f"Cached de-quantized weights to {cache_path!r} for future loads", file=sys.stderr)
    except Exception as e:
        print(f"Warning: failed to write de-quant cache for {source_path!r}: {e}", file=sys.stderr)


def _peek_checkpoint_shapes(path: str):
    """Return (tensor names, {name: torch-order shape}) for a weights file
    without materialising any tensor data. GGUF is read through GGUFReader
    (which memory-maps rather than loads) and safetensors through the lazy
    `safe_open` slice API, so this stays cheap on multi-GB checkpoints."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gguf":
        from gguf import GGUFReader

        reader = GGUFReader(path)
        # GGUF records dimensions in the reverse of torch's order.
        return (
            {t.name for t in reader.tensors},
            {t.name: tuple(int(d) for d in reversed(t.shape)) for t in reader.tensors},
        )

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        names = set(f.keys())
        shapes = {n: tuple(f.get_slice(n).get_shape()) for n in names}
    return names, shapes


def _detect_wan_variant(transformer_path: str) -> str:
    """Identify which Wan architecture `transformer_path` holds, so the right
    config repo and transformer/pipeline classes are used to build it.

    This mirrors diffusers' `infer_diffusers_model_type` (same keys, same
    width thresholds) with one addition: diffusers has no case for the Wan2.2
    TI2V-5B's patch_embedding width of 3072 and silently falls through to its
    14B branch, so that width is claimed here explicitly."""
    names, shapes = _peek_checkpoint_shapes(transformer_path)
    key = next((n for n in names if n.endswith("patch_embedding.weight")), None)
    if key is None:
        raise RuntimeError(
            f"Cannot identify the Wan variant in {transformer_path!r}: no "
            "'patch_embedding.weight' tensor found."
        )

    shape = shapes[key]
    dim, in_channels = shape[0], shape[1]
    is_vace = any("vace_blocks" in n for n in names)

    if is_vace:
        variant = "wan-vace-1.3B" if dim == 1536 else "wan-vace-14B"
    elif dim == 1536:
        variant = "wan-t2v-1.3B"
    elif dim == 3072:
        variant = "wan-ti2v-5B"
    elif dim == 5120 and in_channels == 16:
        variant = "wan-t2v-14B"
    else:
        variant = "wan-i2v-14B"

    print(
        f"Detected Wan variant {variant!r} from {os.path.basename(transformer_path)} "
        f"(patch_embedding {dim}x{in_channels}, vace_blocks={is_vace})",
        file=sys.stderr,
    )
    return variant


def _describe_absent_tensors(transformer_path: str, transformer_cls, base_repo: str) -> str:
    """Name the tensors `transformer_path` lacks relative to what `base_repo`'s
    config says the architecture needs. Only called on the failure path above,
    where the cost of re-reading the checkpoint buys an error a user can act on.
    Degrades to a vague-but-honest phrase if the comparison itself fails."""
    try:
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

        converted = convert_wan_transformer_to_diffusers(
            _load_state_dict_from_file(transformer_path), config=None
        )
        with torch.device("meta"):
            reference = transformer_cls.from_config(
                transformer_cls.load_config(base_repo, subfolder="transformer")
            )
        absent = sorted(set(reference.state_dict()) - set(converted))
    except Exception:
        return "one or more required tensors"
    if not absent:
        return "one or more required tensors"
    return f"{len(absent)} tensor(s) ({', '.join(absent[:5])}{' ...' if len(absent) > 5 else ''})"


def _load_wan_pipeline_from_single_files(
    transformer_path: str,
    text_encoder_override: str = None,
    vae_override: str = None,
):
    """Build a WanPipeline from a loose (bare) transformer weights file
    (`.gguf` or `.safetensors`), with optional sibling (or manually
    overridden) VAE and text-encoder files, pulling any other required
    components (text_encoder/tokenizer/scheduler, and vae/text_encoder if
    not found locally) from WAN_BASE_REPO. Mirrors
    `_load_ltx_pipeline_from_single_files` above, for Wan2.2 instead of
    LTX-Video. No local-components-dir caching exists for Wan — that's an
    LTX-only feature (see LTX_COMPONENTS_DIR)."""
    variant = _detect_wan_variant(transformer_path)
    base_repo, transformer_cls_name, pipeline_cls_name = _WAN_VARIANTS[variant]

    required_symbols = ("AutoencoderKLWan", "GGUFQuantizationConfig", transformer_cls_name, pipeline_cls_name)
    missing = [s for s in required_symbols if not hasattr(diffusers, s)]
    if missing:
        raise RuntimeError(
            f"Cannot load Wan2.2 model from loose weights file {transformer_path!r}: "
            f"installed diffusers ({getattr(diffusers, '__version__', '?')}) is missing "
            f"{', '.join('diffusers.' + s for s in missing)}. Upgrade diffusers to a version "
            "with Wan single-file loading support."
        )

    ext = os.path.splitext(transformer_path)[1].lower()
    if ext not in (".gguf", ".safetensors"):
        raise RuntimeError(
            f"Cannot load Wan2.2 transformer from {transformer_path!r}: unsupported "
            f"extension {ext!r} (expected .gguf or .safetensors)."
        )

    try:
        # Load the text encoder FIRST. transformers' GGUF loader has no lazy
        # path: it dequantizes every tensor eagerly, and `dequantize()` builds
        # a full fp32 array before the fp16 cast, so umt5-xxl's 1.05B-element
        # token_embd alone spikes 4.2 GB on top of the ~9 GB of tensors already
        # converted. Holding the transformer in RAM across that spike is what
        # pushes a 16 GB machine over the commit limit.
        text_encoder_path = text_encoder_override if text_encoder_override else _find_sibling_text_encoder(transformer_path)
        if text_encoder_path:
            if os.path.splitext(text_encoder_path)[1].lower() == ".gguf":
                cached = _try_load_dequant_cache(text_encoder_path)
                if cached is not None:
                    print(f"Loading text encoder from de-quant cache (skipping GGUF conversion): {text_encoder_path!r}", file=sys.stderr)
                    from transformers import UMT5Config, UMT5EncoderModel

                    text_encoder_dir = os.path.dirname(text_encoder_path) or "."
                    config_json = os.path.join(text_encoder_dir, "config.json")
                    config = UMT5Config.from_pretrained(text_encoder_dir) if os.path.isfile(config_json) else UMT5Config(**_UMT5_XXL_CONFIG)
                    with torch.device("meta"):
                        text_encoder = UMT5EncoderModel(config)
                    for key in list(cached.keys()):
                        cached[key] = cached[key].to(torch.float16)
                    text_encoder.load_state_dict(cached, strict=False, assign=True)
                    del cached
                    # `_save_dequant_cache` drops keys that alias another
                    # tensor's storage, so re-establish those aliases here —
                    # otherwise the dropped parameters stay on the meta device.
                    text_encoder.tie_weights()
                else:
                    from transformers import UMT5EncoderModel

                    text_encoder_dir = os.path.dirname(text_encoder_path) or "."
                    text_encoder = UMT5EncoderModel.from_pretrained(
                        text_encoder_dir, gguf_file=os.path.basename(text_encoder_path), torch_dtype=torch.float16,
                    )
                    _save_dequant_cache(text_encoder_path, text_encoder.state_dict())
            else:
                text_encoder = _load_umt5_text_encoder_from_state_dict(text_encoder_path)
        else:
            text_encoder = None

        # Pass the config explicitly rather than letting `from_single_file`
        # infer it: diffusers' `infer_diffusers_model_type` has no case for the
        # Wan2.2 TI2V-5B's patch_embedding width of 3072 and silently falls
        # through to its 14B branch, building a dim-5120 module that the real
        # checkpoint cannot populate. `_detect_wan_variant` covers that width
        # alongside the ones diffusers does handle.
        transformer_cls = getattr(diffusers, transformer_cls_name)
        single_file_kwargs = {
            "config": base_repo,
            "subfolder": "transformer",
            "torch_dtype": torch.float16,
        }
        if ext == ".gguf":
            # No de-quant cache for the transformer, unlike the text encoder
            # above: diffusers keeps GGUF weights packed and de-quantizes them
            # lazily inside each GGUFLinear forward, so nothing is converted at
            # load time and there is no work for a cache to skip. Its
            # `state_dict()` is still the quantized blocks (a cache write here
            # produced a 3.83 GB file, matching the GGUF rather than the ~10 GB
            # an fp16 copy would need), which would not reload into the plain
            # fp16 module that `from_config` builds.
            single_file_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                compute_dtype=torch.float16
            )
        try:
            transformer = transformer_cls.from_single_file(transformer_path, **single_file_kwargs)
        except NotImplementedError as e:
            # `from_single_file` raises for shapes that conflict but not for keys
            # the checkpoint simply lacks — those parameters stay on the meta
            # device until accelerate's dispatch trips over them, reporting only
            # "Cannot copy out of meta tensor; no data!" and naming nothing.
            # Work out which tensors are actually absent and say so.
            if "meta tensor" not in str(e):
                raise
            raise RuntimeError(
                f"{os.path.basename(transformer_path)} is incomplete: it is missing "
                f"{_describe_absent_tensors(transformer_path, transformer_cls, base_repo)} "
                f"that the {variant} architecture requires. Re-download the file or use a "
                "different conversion of this model."
            ) from e

        vae_path = vae_override if vae_override else _find_sibling_vae(transformer_path)
        vae = None
        if vae_path:
            # Unlike LTX's VAE conversion, diffusers' Wan VAE conversion
            # (convert_wan_vae_to_diffusers) matches keys like
            # "decoder.middle.0.residual.0.gamma" with no "vae." prefix, so
            # a standalone VAE-only checkpoint's keys are used as-is here —
            # no prefixing trick needed.
            vae_state_dict = _load_state_dict_from_file(vae_path)
            vae = _load_wan_vae_from_state_dict(vae_state_dict)

        pipeline_kwargs = {"transformer": transformer, "torch_dtype": torch.float16}
        if vae is not None:
            pipeline_kwargs["vae"] = vae
        if text_encoder is not None:
            pipeline_kwargs["text_encoder"] = text_encoder

        pipe = getattr(diffusers, pipeline_cls_name).from_pretrained(base_repo, **pipeline_kwargs)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Wan pipeline ({variant}) from local weights {transformer_path!r} "
            f"(base repo {base_repo!r} used for missing components): {e}"
        ) from e

    return pipe


def _resolve_pipeline_class(model_path: str):
    """Work out which diffusers pipeline class `model_path` will produce,
    reading only headers — `model_index.json` for a packaged pipeline, the GGUF
    metadata block for a loose checkpoint. No weights are touched, so this
    costs milliseconds and can run before the port is even bound.

    Returns None when the class can't be determined, in which case `/schema`
    falls back to advertising the full PARAM_SCHEMA."""
    try:
        index_path = os.path.join(model_path, "model_index.json") if os.path.isdir(model_path) else None
        if index_path and os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                return getattr(diffusers, json.load(f).get("_class_name", ""), None)

        loose_path = model_path
        if os.path.isdir(model_path):
            loose_path = _find_loose_main_weights(model_path)
        if not loose_path or os.path.splitext(loose_path)[1].lower() not in (".gguf", ".safetensors"):
            return None

        if os.path.splitext(loose_path)[1].lower() == ".gguf":
            architecture = _detect_gguf_architecture(loose_path)
        else:
            architecture = "ltxv"

        if architecture == "wan":
            return getattr(diffusers, _WAN_VARIANTS[_detect_wan_variant(loose_path)][2], None)
        if architecture in ("ltxv", "ltx-video"):
            return getattr(diffusers, "LTXPipeline", None)
    except Exception as e:
        print(f"Warning: could not resolve the pipeline class for /schema: {e}", file=sys.stderr)
    return None


def load_pipeline(model_path: str, text_encoder_override: str = None, vae_override: str = None):
    is_full_pipeline_dir = os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "model_index.json"))

    loose_transformer_path = None
    if not is_full_pipeline_dir:
        if os.path.isfile(model_path) and os.path.splitext(model_path)[1].lower() in (".gguf", ".safetensors"):
            loose_transformer_path = model_path
        elif os.path.isdir(model_path):
            loose_transformer_path = _find_loose_main_weights(model_path)

    if loose_transformer_path is not None:
        # Only .gguf files carry architecture metadata we can inspect; a bare
        # .safetensors state dict has no such marker, so it keeps defaulting
        # to the (only other supported) LTX-Video path.
        if os.path.splitext(loose_transformer_path)[1].lower() == ".gguf":
            architecture = _detect_gguf_architecture(loose_transformer_path)
        else:
            architecture = "ltxv"

        if architecture == "wan":
            pipe = _load_wan_pipeline_from_single_files(loose_transformer_path, text_encoder_override, vae_override)
        elif architecture in ("ltxv", "ltx-video"):
            pipe = _load_ltx_pipeline_from_single_files(loose_transformer_path, text_encoder_override, vae_override)
        else:
            raise RuntimeError(
                f"Cannot load loose weights file {loose_transformer_path!r}: unrecognized GGUF "
                f"architecture {architecture!r} (only 'ltxv' and 'wan' are supported for loose "
                "single-file loading)."
            )
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

    _pin_text_encoder_to_cpu_if_oversized(pipe)

    return pipe


def _pin_text_encoder_to_cpu_if_oversized(pipe):
    """`enable_model_cpu_offload` moves each component onto the GPU whole when
    its turn comes, and the text encoder's turn is first. UMT5-XXL is 11.4 GB
    in fp16, so on an 8 GB card encoding the prompt OOMs before a single
    denoise step runs. When the encoder cannot fit, strip its offload hook and
    pin it to the CPU; `_precompute_prompt_embeds` then encodes there and hands
    the pipeline `prompt_embeds`, so the encoder is never asked for a forward
    pass on the GPU. That costs one CPU pass over a few hundred tokens, against
    a denoise loop that then gets the whole card to itself."""
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None or DEVICE != "cuda" or not hasattr(pipe, "encode_prompt"):
        return
    if "prompt_embeds" not in inspect.signature(pipe.__call__).parameters:
        return

    needed = sum(p.numel() * p.element_size() for p in text_encoder.parameters())
    free_vram = torch.cuda.mem_get_info()[0]
    if needed <= free_vram * 0.8:
        return

    _strip_offload_hook_and_pin_to_cpu(text_encoder)
    pipe._dispos_cpu_text_encoder = True
    print(
        "Text encoder needs %.1f GB but only %.1f GB of VRAM is free: pinning it to "
        "the CPU and pre-computing prompt embeddings there." % (needed / 2**30, free_vram / 2**30),
        file=sys.stderr,
    )


def _strip_offload_hook_and_pin_to_cpu(module) -> None:
    """Drop any accelerate offload hook on `module` and move it to the CPU, so
    a forward pass runs there instead of relocating the weights to the GPU."""
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(module, recurse=True)
    except Exception:
        pass
    module.to("cpu")


def _precompute_prompt_embeds(kwargs: dict, accepted) -> None:
    """For a CPU-pinned text encoder (see `_pin_text_encoder_to_cpu_if_oversized`),
    encode on the CPU here and swap `prompt`/`negative_prompt` for the resulting
    embeddings. `encode_prompt` returns early when it is handed embeddings, so
    the pipeline never invokes the text encoder and never moves it to the GPU."""
    if not getattr(PIPE, "_dispos_cpu_text_encoder", False):
        return
    if "prompt_embeds" not in accepted or not kwargs.get("prompt"):
        return

    # Every `__call__` ends in `maybe_free_model_hooks`, which re-runs
    # `enable_model_cpu_offload` and so re-attaches the offload hook that the
    # initial pin removed. Without re-stripping it here, the second generation
    # would relocate the text encoder to the GPU and OOM exactly like the first.
    _strip_offload_hook_and_pin_to_cpu(PIPE.text_encoder)

    encode_params = inspect.signature(PIPE.encode_prompt).parameters
    encode_kwargs = {"prompt": kwargs["prompt"], "device": torch.device("cpu")}
    for name in ("negative_prompt", "max_sequence_length"):
        if name in encode_params and kwargs.get(name) is not None:
            encode_kwargs[name] = kwargs[name]

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = PIPE.encode_prompt(**encode_kwargs)

    kwargs.pop("prompt", None)
    kwargs.pop("negative_prompt", None)
    kwargs["prompt_embeds"] = prompt_embeds.to(DEVICE)
    if negative_prompt_embeds is not None and "negative_prompt_embeds" in accepted:
        kwargs["negative_prompt_embeds"] = negative_prompt_embeds.to(DEVICE)


def main():
    global PIPE, DEVICE, LTX_COMPONENTS_DIR, PIPE_CLASS, PIPE_LOAD_ERROR

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ltx-components-dir", default=None)
    parser.add_argument("--text-encoder-path", default=None, help="Manual override for the LTX-Video text encoder (gguf/safetensors/ckpt/pt/bin)")
    parser.add_argument("--vae-path", default=None, help="Manual override for the LTX-Video VAE (gguf/safetensors/ckpt/pt/bin)")
    args = parser.parse_args()

    LTX_COMPONENTS_DIR = args.ltx_components_dir

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve the pipeline class from headers alone, before any weights are
    # read, so `/schema` can answer as soon as the port is bound.
    PIPE_CLASS = _resolve_pipeline_class(args.model_path)

    # Refuse to share a port with an orphaned server from an earlier daemon run —
    # on Windows SO_REUSEADDR would let the bind succeed and split requests between
    # the two processes, so requests silently land on the stale one.
    class ExclusiveHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = False

    server = ExclusiveHTTPServer(("127.0.0.1", args.port), Handler)

    # Load the weights on a background thread and signal READY now rather than
    # after. The daemon spawns this process to answer `/schema` too, so loading
    # first meant the studio couldn't render any parameter fields until a
    # generation had forced the load to complete. `/schema` is served off
    # PIPE_CLASS meanwhile; `/generate` waits on PIPE_READY.
    def _load_in_background():
        global PIPE, PIPE_LOAD_ERROR
        try:
            print(f"Loading diffusers video pipeline from {args.model_path}...", file=sys.stderr)
            PIPE = load_pipeline(args.model_path, args.text_encoder_path, args.vae_path)
            print(f"diffusers video pipeline loaded on device: {DEVICE}", file=sys.stderr)
        except BaseException as e:
            import traceback

            PIPE_LOAD_ERROR = str(e)
            traceback.print_exc(file=sys.stderr)
        finally:
            PIPE_READY.set()

    threading.Thread(target=_load_in_background, name="pipeline-loader", daemon=True).start()

    print("READY", flush=True)
    sys.stdout.flush()

    server.serve_forever()


if __name__ == "__main__":
    main()
