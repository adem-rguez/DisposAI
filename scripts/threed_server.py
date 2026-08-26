"""
3D mesh generation HTTP server.

Spawned as a subprocess by the Rust `mesh-backend` crate. Selects a
text/image-to-3D adapter from the model path at startup and serves
generation requests over a stdlib HTTP server.

Supported adapters:
  - Shap-E (fully-supported baseline): `diffusers`' `ShapEPipeline` (text) /
    `ShapEImg2ImgPipeline` (image) + `trimesh`. Guaranteed to work if
    `diffusers` and `trimesh` are installed.
  - TripoSR, Stable-Fast-3D (SF3D), TRELLIS, Hunyuan3D-2, Point-E, VGGT:
    adapters that follow their documented package APIs (`tsr`, `sf3d`,
    `trellis`, `hy3dgen`, `point_e`, `vggt`). Guarded by try/except; if the extra package isn't
    installed the server still serves a placeholder cube instead of failing
    to start. See `scripts/setup_3d_env.py` for how to install the heavy
    native deps these need into a dedicated Python 3.11 venv.
  - InstantMesh, Wonder3D/SV3D/Zero123, CRM/LGM: no pip-installable package
    exists upstream today, so these always resolve to the placeholder cube.

The server NEVER crashes a request: every /generate call either returns a
valid mesh or a clean `{"error": ...}` JSON body.

Adapter registry (modular by design):
  Every adapter is a `MeshAdapter` subclass declaring `name`, `input_kinds`,
  `output_formats`, and `match_tokens` (lowercase substrings of the combined
  "parent dir + file name" string that select it) as class attributes.
  `ADAPTER_REGISTRY` is the single ordered list of these classes; order is
  precedence (first token match wins), so e.g. Point-E/Shap-E's specific
  tokens are listed ahead of any adapter that might loosely match "3d"/"shape".
  `select_adapter()` just walks the registry, finds the first class whose
  `match_tokens` match, instantiates it, and uses it if `is_available()`
  else falls back to `CubeAdapter(input_kinds=<that class's input_kinds>)`.

  To add a new architecture: write one `MeshAdapter` subclass with its
  `match_tokens` + `generate()`, add it to `ADAPTER_REGISTRY` — nothing else
  in this file needs to change. IMPORTANT: `MESH3D_NAME_TOKENS` and
  `detect_mesh_input_kinds()` in `crates/daemon-core/src/http.rs` do their
  own independent name-matching on the Rust side (to report input kinds to
  the UI before this Python process starts) and must be kept in sync with
  any new `match_tokens` added here.

Usage:
    python threed_server.py --model-path <path/to/model> --port <port>

Protocol:
    POST /generate
        body: {"prompt": "...", "images": ["<base64>", ...],
                "input_kind": "text"|"image"|"multi_image", "steps": 64,
                "guidance_scale": 15.0, "seed": null, "output_format": "glb",
                "texture": true, "foreground_ratio": 0.85}
        200: {"mesh_base64": "...", "format": "glb"}
        4xx/5xx: {"error": "..."}
    GET /capabilities -> {"input_kinds": [...], "pipeline": "...", "output_formats": [...]}
    GET /schema -> {"<adapter_name>": {"params": [...PARAM_SPEC...]}, ...} for every
        registered adapter (see `MeshAdapter.PARAM_SPEC` / `resolve_params()`).
    GET /health -> {"status": "ok"}

Prints "READY" to stdout once the adapter is selected and the server is
listening. The Rust backend watches for this line.
"""

import argparse
import atexit
import base64
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Import torch up front: creating a CUDA session with a different DLL-loading
# order than torch expects can fail with WinError 127. Loading torch first
# and injecting the nvidia package DLL directories wins. Mirrors
# diffusers_server.py / kokoro_tts_server.py / hf_transformers_server.py's
# import ordering.
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

if sys.platform == "win32":
    import importlib.util as _ilu
    for _mod in ("nvidia.cu13", "nvidia.cudnn"):
        # find_spec raises (not returns None) when the parent `nvidia` package
        # is absent — which is the norm on Windows, where torch's CUDA wheels
        # bundle their DLLs under torch/lib instead of separate nvidia-* pip
        # packages. Treat that as "nothing to inject" rather than crashing.
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
    import trimesh
except ImportError as e:
    print(f"Failed to import trimesh: {e}", file=sys.stderr)
    trimesh = None

try:
    from PIL import Image
except ImportError as e:
    print(f"Failed to import PIL: {e}", file=sys.stderr)
    Image = None

ADAPTER = None

PROGRESS_LOCK = threading.Lock()
PROGRESS = {"status": "idle"}


def _progress_reset(total: int = 0):
    with PROGRESS_LOCK:
        PROGRESS.clear()
        PROGRESS.update({
            "job_id": "",
            "modality": "mesh",
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


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------

def export_mesh(mesh, output_format):
    """Export a `trimesh.Trimesh` to bytes in `output_format` (glb/obj/ply/stl).
    Returns (bytes, normalized_format)."""
    fmt = (output_format or "glb").lower()
    if fmt not in ("glb", "obj", "ply", "stl"):
        fmt = "glb"
    data = mesh.export(file_type=fmt)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data, fmt


def _cube_mesh():
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def _as_numpy(x):
    """Detach/move-to-cpu/convert-to-numpy, tolerating plain numpy arrays."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x


def _to_trimesh(obj):
    """Best-effort conversion of a model's mesh output into a single
    `trimesh.Trimesh`. Handles:
      - an already-built `trimesh.Trimesh` (passthrough)
      - a `trimesh.Scene` (flattened/concatenated into one mesh)
      - a `(verts, faces)` tuple
      - any object exposing `.vertices`/`.faces` (or `.verts`/`.faces`) as
        torch tensors or numpy arrays, carrying vertex colors/uv if present
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj

    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError("trimesh.Scene has no geometry to export")
        return geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)

    vertex_colors = None
    uv = None
    if isinstance(obj, tuple) and len(obj) == 2:
        verts, faces = obj
    else:
        verts = getattr(obj, "vertices", None)
        if verts is None:
            verts = getattr(obj, "verts", None)
        faces = getattr(obj, "faces", None)
        if verts is None or faces is None:
            raise TypeError(f"Don't know how to convert {type(obj)!r} to a trimesh.Trimesh")
        vertex_colors = getattr(obj, "vertex_colors", None)
        uv = getattr(obj, "uv", None)

    verts = _as_numpy(verts)
    faces = _as_numpy(faces)

    kwargs = {}
    if vertex_colors is not None:
        kwargs["vertex_colors"] = _as_numpy(vertex_colors)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, **kwargs)

    if uv is not None:
        try:
            from trimesh.visual import TextureVisuals
            mesh.visual = TextureVisuals(uv=_as_numpy(uv))
        except Exception:
            pass  # texture is a nice-to-have; keep the untextured mesh on failure

    return mesh


# ---------------------------------------------------------------------------
# Adapter parameter specs
# ---------------------------------------------------------------------------
#
# Each adapter declares `PARAM_SPEC`: a list of dicts describing exactly the
# params it reads out of the raw `/generate` JSON body, e.g.:
#   {"name": "steps", "type": "int", "default": 64, "min": 1, "max": 150}
# `type` is one of "float"|"int"|"bool"|"str"|"enum" ("enum" uses "choices").
# `resolve_params()` applies default + type coercion + min/max/choices
# clamping, so the spec is the actual source of truth for each adapter's
# `generate()`, not just documentation. `GET /schema` serializes these specs
# for every registered adapter.

def _coerce_param(spec_item, raw_value):
    """Apply one PARAM_SPEC item's type coercion, default, and min/max/choices
    clamping to a single raw value pulled from the request body."""
    ptype = spec_item.get("type", "str")
    default = spec_item.get("default")

    if raw_value is None:
        value = default
    else:
        try:
            if ptype == "float":
                value = float(raw_value)
            elif ptype == "int":
                value = int(raw_value)
            elif ptype == "bool":
                value = bool(raw_value)
            else:
                value = raw_value
        except (TypeError, ValueError):
            value = default

    if ptype in ("float", "int") and value is not None:
        if spec_item.get("min") is not None:
            value = max(spec_item["min"], value)
        if spec_item.get("max") is not None:
            value = min(spec_item["max"], value)
    if ptype == "enum" and spec_item.get("choices") and value not in spec_item["choices"]:
        value = default

    return value


def resolve_params(spec, params):
    """Resolve a `{name: value}` dict from the raw `/generate` params according
    to `spec` (a PARAM_SPEC list)."""
    return {item["name"]: _coerce_param(item, params.get(item["name"])) for item in spec}


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class MeshAdapter:
    name = "base"
    input_kinds = ["image"]
    output_formats = ["glb", "obj", "ply", "stl"]
    # Lowercase substrings of the combined "parent dir + file name" string
    # that select this adapter in `ADAPTER_REGISTRY`. See module docstring.
    match_tokens = []
    # Declared generation params this adapter reads via `resolve_params()`.
    # See "Adapter parameter specs" above.
    PARAM_SPEC = []

    def __init__(self, model_path):
        self.model_path = model_path

    def is_available(self):
        return True

    def resolve_params(self, params):
        return resolve_params(self.PARAM_SPEC, params)

    def generate(self, input_kind, prompt, images, params):
        raise NotImplementedError


class CubeAdapter(MeshAdapter):
    """Fallback adapter: always available (given `trimesh`), returns a unit
    cube. Used when the ideal adapter for a model can't be constructed
    (missing deps / unsupported architecture / no adapter implemented)."""
    name = "cube-placeholder"

    def __init__(self, input_kinds=None):
        if input_kinds:
            self.input_kinds = input_kinds

    def is_available(self):
        return trimesh is not None

    def generate(self, input_kind, prompt, images, params):
        return _cube_mesh()


class ShapEAdapter(MeshAdapter):
    """Fully-supported baseline: `diffusers` ShapEPipeline (text) /
    ShapEImg2ImgPipeline (image)."""
    name = "shap-e"
    input_kinds = ["text", "image"]
    match_tokens = ["shap-e", "shap_e", "shape"]
    PARAM_SPEC = [
        {"name": "steps", "type": "int", "default": 64, "min": 1, "max": 150},
        {"name": "guidance_scale", "type": "float", "default": 15.0, "min": 0.0, "max": 30.0},
        {"name": "seed", "type": "int", "default": None},
    ]

    def __init__(self, model_path):
        self.model_path = model_path
        self._text_pipe = None
        self._img_pipe = None

    def is_available(self):
        try:
            import diffusers  # noqa: F401
            return trimesh is not None and torch is not None
        except ImportError:
            return False

    def _device(self):
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def _dtype(self):
        return torch.float16 if self._device() == "cuda" else torch.float32

    def _resolve_repo(self, default_repo):
        if os.path.isdir(self.model_path) and os.path.exists(os.path.join(self.model_path, "model_index.json")):
            return self.model_path
        parent = os.path.dirname(self.model_path)
        if parent and os.path.exists(os.path.join(parent, "model_index.json")):
            return parent
        return default_repo

    def _load_text_pipe(self):
        if self._text_pipe is None:
            from diffusers import ShapEPipeline
            repo = self._resolve_repo("openai/shap-e")
            self._text_pipe = ShapEPipeline.from_pretrained(repo, torch_dtype=self._dtype()).to(self._device())
        return self._text_pipe

    def _load_img_pipe(self):
        if self._img_pipe is None:
            from diffusers import ShapEImg2ImgPipeline
            repo = self._resolve_repo("openai/shap-e-img2img")
            self._img_pipe = ShapEImg2ImgPipeline.from_pretrained(repo, torch_dtype=self._dtype()).to(self._device())
        return self._img_pipe

    def generate(self, input_kind, prompt, images, params):
        resolved = self.resolve_params(params)
        steps = resolved["steps"]
        guidance_scale = resolved["guidance_scale"]
        seed = resolved["seed"]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device()).manual_seed(int(seed))

        _progress_update(phase="generating", status="running", step=0, total=steps)

        def _cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
            _progress_update(step=step_index + 1, total=steps)
            return callback_kwargs

        def _cb_legacy(step, timestep, latents):
            _progress_update(step=step + 1, total=steps)

        if input_kind == "text":
            pipe = self._load_text_pipe()
            try:
                result = pipe(
                    prompt or "",
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=generator,
                    output_type="mesh",
                    callback_on_step_end=_cb_on_step_end,
                )
            except TypeError:
                result = pipe(
                    prompt or "",
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=generator,
                    output_type="mesh",
                    callback=_cb_legacy,
                    callback_steps=1,
                )
        else:
            if not images:
                raise ValueError("Shap-E image input requires at least one image")
            pipe = self._load_img_pipe()
            try:
                result = pipe(
                    images[0],
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=generator,
                    output_type="mesh",
                    callback_on_step_end=_cb_on_step_end,
                )
            except TypeError:
                result = pipe(
                    images[0],
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=generator,
                    output_type="mesh",
                    callback=_cb_legacy,
                    callback_steps=1,
                )

        mesh_output = result.images[0]
        return _to_trimesh(mesh_output)


class TripoSRAdapter(MeshAdapter):
    """Best-effort adapter for stabilityai/TripoSR, following its documented
    `tsr` package API (https://github.com/VAST-AI-Research/TripoSR)."""
    name = "triposr"
    input_kinds = ["image"]
    match_tokens = ["triposr", "tripo"]
    PARAM_SPEC = [
        {"name": "foreground_ratio", "type": "float", "default": 0.85, "min": 0.0, "max": 1.0},
    ]

    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None

    def is_available(self):
        try:
            from tsr.system import TSR  # noqa: F401
            return trimesh is not None
        except ImportError:
            return False

    @staticmethod
    def _remap_ckpt(ckpt, model):
        """Remap HF-ViT checkpoint keys to match the tsr package's custom ViT."""
        model_keys = set(model.state_dict().keys())
        if all(k in model_keys for k in ckpt.keys()):
            return ckpt
        remap = {}
        for k, v in ckpt.items():
            nk = k
            # encoder.layer.N -> layers.N
            nk = nk.replace(".encoder.layer.", ".layers.")
            # attention.attention.query -> attention.q_proj
            nk = nk.replace(".attention.attention.query.", ".attention.q_proj.")
            nk = nk.replace(".attention.attention.key.", ".attention.k_proj.")
            nk = nk.replace(".attention.attention.value.", ".attention.v_proj.")
            nk = nk.replace(".attention.output.dense.", ".attention.o_proj.")
            # intermediate.dense -> mlp.fc1, output.dense -> mlp.fc2
            nk = nk.replace(".intermediate.dense.", ".mlp.fc1.")
            nk = nk.replace(".output.dense.", ".mlp.fc2.")
            remap[nk] = v
        return remap

    def _load(self):
        if self._model is None:
            from tsr.system import TSR
            from omegaconf import OmegaConf
            repo = self.model_path if os.path.isdir(self.model_path) else "stabilityai/TripoSR"
            config_name = "config.yaml"
            weight_name = "model.ckpt"
            if os.path.isdir(repo):
                if not os.path.isfile(os.path.join(repo, config_name)) and os.path.isfile(os.path.join(repo, "resolved_config.json")):
                    config_name = "resolved_config.json"
                if not os.path.isfile(os.path.join(repo, weight_name)) and os.path.isfile(os.path.join(repo, "model.safetensors")):
                    weight_name = "model.safetensors"
            if os.path.isdir(repo):
                cfg_path = os.path.join(repo, config_name)
                weight_path = os.path.join(repo, weight_name)
            else:
                from huggingface_hub import hf_hub_download
                cfg_path = hf_hub_download(repo_id=repo, filename=config_name)
                weight_path = hf_hub_download(repo_id=repo, filename=weight_name)
            cfg = OmegaConf.load(cfg_path)
            OmegaConf.resolve(cfg)
            self._model = TSR(cfg)
            if weight_name.endswith(".safetensors"):
                from safetensors.torch import load_file as load_safetensors
                ckpt = load_safetensors(weight_path)
            else:
                ckpt = torch.load(weight_path, map_location="cpu")
            ckpt = self._remap_ckpt(ckpt, self._model)
            self._model.load_state_dict(ckpt)
            self._model.renderer.set_chunk_size(8192)
            if torch is not None and torch.cuda.is_available():
                self._model.to("cuda")
        return self._model

    def generate(self, input_kind, prompt, images, params):
        if not images:
            raise ValueError("TripoSR requires an image input")
        model = self._load()
        image = images[0].convert("RGB")
        foreground_ratio = self.resolve_params(params)["foreground_ratio"]
        try:
            from tsr.utils import remove_background, resize_foreground
            import rembg
            image = remove_background(image, rembg.new_session())
            image = resize_foreground(image, foreground_ratio)
        except ImportError:
            pass  # fall back to using the raw image unsegmented
        if image.mode != "RGB":
            image = image.convert("RGB")

        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            scene_codes = model([image], device=device)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True)
        return _to_trimesh(meshes[0])


class SF3DAdapter(MeshAdapter):
    """Best-effort adapter for stabilityai/stable-fast-3d, following its
    documented `sf3d` package API."""
    name = "stable-fast-3d"
    input_kinds = ["image"]
    match_tokens = ["sf3d", "stable-fast-3d", "stable_fast_3d"]
    PARAM_SPEC = [
        {"name": "foreground_ratio", "type": "float", "default": 0.85, "min": 0.0, "max": 1.0},
    ]

    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None

    def is_available(self):
        try:
            from sf3d.system import SF3D  # noqa: F401
            return trimesh is not None
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            from sf3d.system import SF3D
            repo = self.model_path if os.path.isdir(self.model_path) else "stabilityai/stable-fast-3d"
            self._model = SF3D.from_pretrained(repo, config_name="config.yaml", weight_name="model.safetensors")
            if torch is not None and torch.cuda.is_available():
                self._model.to("cuda")
            self._model.eval()
        return self._model

    def generate(self, input_kind, prompt, images, params):
        if not images:
            raise ValueError("Stable Fast 3D requires an image input")
        model = self._load()
        image = images[0].convert("RGBA")
        foreground_ratio = self.resolve_params(params)["foreground_ratio"]
        try:
            import rembg
            from sf3d.utils import remove_background, resize_foreground
            image = remove_background(image, rembg.new_session())
            image = resize_foreground(image, foreground_ratio)
        except ImportError:
            pass
        return self._generate_vertex_colored(model, image)

    @staticmethod
    def _generate_vertex_colored(model, image):
        """Extract mesh geometry and query per-vertex colors from the triplane,
        bypassing UV unwrapping and texture baking entirely."""
        import numpy as np
        from contextlib import nullcontext
        from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else nullcontext():
                mask_cond, rgb_cond = model.prepare_image(image)
                c2w_cond = default_cond_c2w(model.cfg.default_distance).to(model.device)
                intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
                    model.cfg.default_fovy_deg, model.cfg.cond_image_size, model.cfg.cond_image_size,
                )
                batch = {
                    "rgb_cond": rgb_cond,
                    "mask_cond": mask_cond,
                    "c2w_cond": c2w_cond.view(1, 1, 4, 4),
                    "intrinsic_cond": intrinsic.to(model.device).view(1, 1, 3, 3),
                    "intrinsic_normed_cond": intrinsic_normed_cond.to(model.device).view(1, 1, 3, 3),
                }
                batch["rgb_cond"] = model.image_processor(batch["rgb_cond"], model.cfg.cond_image_size)
                batch["mask_cond"] = model.image_processor(batch["mask_cond"], model.cfg.cond_image_size)
                scene_codes, _ = model.get_scene_codes(batch)
                meshes = model.triplane_to_meshes(scene_codes)
                mesh = meshes[0]
                if mesh.v_pos.shape[0] == 0:
                    return trimesh.Trimesh()
                tri_query = model.query_triplane(mesh.v_pos.unsqueeze(0), scene_codes[0:1])[0]
                decoded = model.decoder(tri_query, exclude=["density", "vertex_offset"])
                colors = decoded["features"].squeeze(0)
                colors = colors.float().clamp(0, 1).cpu().numpy()
                colors_uint8 = (colors * 255).astype(np.uint8)
                if colors_uint8.shape[-1] == 3:
                    alpha = np.full((colors_uint8.shape[0], 1), 255, dtype=np.uint8)
                    colors_uint8 = np.concatenate([colors_uint8, alpha], axis=-1)

        verts = mesh.v_pos.float().cpu().numpy()
        faces = mesh.t_pos_idx.cpu().numpy()
        tmesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors_uint8)
        rot = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        tmesh.apply_transform(rot)
        tmesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0]))
        tmesh.invert()
        return tmesh


class VggtAdapter(MeshAdapter):
    """Adapter for facebook/vggt-1b, following its documented `vggt` package
    API (https://github.com/facebookresearch/vggt). VGGT is a feed-forward
    multi-view reconstruction model: it predicts per-image depth maps (or
    point maps) + camera poses but does NOT emit a mesh directly, so this
    adapter fuses the predicted per-image point maps into a single point
    cloud and runs Poisson surface reconstruction (via `open3d`) to obtain a
    watertight mesh with faces. Text input isn't supported."""
    name = "vggt"
    input_kinds = ["image", "multi_image"]
    match_tokens = ["vggt", "vggt-1b", "vggt_1b"]
    PARAM_SPEC = [
        {"name": "conf_threshold", "type": "float", "default": 50.0, "min": 0.0, "max": 100.0},
        {"name": "poisson_depth", "type": "int", "default": 9, "min": 4, "max": 12},
    ]

    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None

    def is_available(self):
        try:
            from vggt.models.vggt import VGGT  # noqa: F401
            import open3d  # noqa: F401
            return trimesh is not None
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            from vggt.models.vggt import VGGT
            repo = self.model_path if os.path.isdir(self.model_path) else "facebook/VGGT-1B"
            self._model = VGGT.from_pretrained(repo)
            self._model.eval()
            if torch is not None and torch.cuda.is_available():
                self._model.to("cuda")
        return self._model

    def generate(self, input_kind, prompt, images, params):
        if not images:
            raise ValueError("VGGT requires at least one image input")
        import numpy as np
        import open3d as o3d
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        resolved = self.resolve_params(params)
        conf_threshold = resolved["conf_threshold"]
        poisson_depth = resolved["poisson_depth"]

        model = self._load()
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # load_and_preprocess_images expects file paths; VGGT's own resize/crop
        # logic (pad to 14-divisible dims, center crop) is easiest to reuse by
        # writing the decoded PIL images to a temp dir and pointing it there.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = []
            for i, img in enumerate(images):
                path = os.path.join(tmp_dir, f"{i:03d}.png")
                img.convert("RGB").save(path)
                image_paths.append(path)
            image_tensor = load_and_preprocess_images(image_paths).to(device)

        with torch.no_grad():
            autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else torch.autocast(device_type="cpu", enabled=False)
            with autocast_ctx:
                batched = image_tensor[None]
                predictions = model(batched)
                pose_enc = predictions["pose_enc"]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
                depth_map = predictions["depth"]
                depth_conf = predictions["depth_conf"]

        world_points = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )
        depth_conf = _as_numpy(depth_conf.squeeze(0))
        colors = (_as_numpy(image_tensor).transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
        extrinsic_np = _as_numpy(extrinsic.squeeze(0))

        threshold_value = np.percentile(depth_conf, conf_threshold) if depth_conf.size else 0.0
        mask = depth_conf >= threshold_value

        # Orient normals per-view towards that view's own camera center before
        # merging. A single global tangent-plane fit over a sparse, multi-view
        # (or single-view "front only") point cloud tends to produce
        # inconsistently-oriented normals, which makes Poisson reconstruction
        # emit a broken/one-sided shell instead of a closed surface. Each
        # view's points are, by construction, front-facing to its own camera,
        # so orienting per-view against the known camera center is far more
        # reliable.
        all_points, all_colors, all_normals = [], [], []
        for i in range(world_points.shape[0]):
            view_mask = mask[i]
            if not np.any(view_mask):
                continue
            view_points = world_points[i][view_mask].reshape(-1, 3).astype(np.float64)
            view_colors = colors[i][view_mask].reshape(-1, 3).astype(np.float64) / 255.0
            if view_points.shape[0] < 3:
                continue
            r = extrinsic_np[i][:, :3]
            t = extrinsic_np[i][:, 3]
            cam_center = -r.T @ t  # extrinsic is world->camera [R|t]; center = -R^T t
            view_pcd = o3d.geometry.PointCloud()
            view_pcd.points = o3d.utility.Vector3dVector(view_points)
            view_pcd.estimate_normals()
            view_pcd.orient_normals_towards_camera_location(cam_center)
            all_points.append(view_points)
            all_colors.append(view_colors)
            all_normals.append(np.asarray(view_pcd.normals))

        if not all_points:
            raise ValueError("VGGT produced too few confident 3D points to reconstruct a mesh")

        points = np.concatenate(all_points, axis=0)
        point_colors = np.concatenate(all_colors, axis=0)
        normals = np.concatenate(all_normals, axis=0)

        if points.shape[0] < 4:
            raise ValueError("VGGT produced too few confident 3D points to reconstruct a mesh")

        # VGGT/OpenCV camera convention is X-right, Y-down, Z-forward
        # (into the scene). Mesh viewers/exporters (trimesh, glTF, etc.)
        # expect Y-up, Z-toward-viewer, which is a 180-degree rotation
        # about X (negate Y and Z). This is a proper rotation (det=+1), so
        # face winding/normals stay consistent and no extra invert() is
        # needed.
        axis_fix = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        points = points @ axis_fix.T
        normals = normals @ axis_fix.T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        densities = np.asarray(densities)
        low_density_mask = densities < np.quantile(densities, 0.05)
        o3d_mesh.remove_vertices_by_mask(low_density_mask)

        verts = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        vertex_colors = None
        if o3d_mesh.has_vertex_colors():
            vertex_colors = (np.asarray(o3d_mesh.vertex_colors) * 255.0).astype(np.uint8)
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)


class TrellisAdapter(MeshAdapter):
    """Guarded shell for microsoft/TRELLIS. Calls the official pipeline API
    when the `trellis` package is importable, otherwise the caller falls back
    to `CubeAdapter`."""
    name = "trellis"
    input_kinds = ["text", "image", "multi_image"]
    match_tokens = ["trellis"]
    PARAM_SPEC = [
        {"name": "seed", "type": "int", "default": 0},
    ]

    def __init__(self, model_path):
        self.model_path = model_path
        self._pipeline = None
        self._text_pipeline = None

    def is_available(self):
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: F401
            return trimesh is not None
        except ImportError:
            return False

    def _load(self):
        if self._pipeline is None:
            from trellis.pipelines import TrellisImageTo3DPipeline
            repo = self.model_path if os.path.isdir(self.model_path) else "microsoft/TRELLIS-image-large"
            self._pipeline = TrellisImageTo3DPipeline.from_pretrained(repo)
            if torch is not None and torch.cuda.is_available():
                self._pipeline.cuda()
        return self._pipeline

    def _load_text(self):
        if self._text_pipeline is None:
            try:
                from trellis.pipelines import TrellisTextTo3DPipeline
            except ImportError as e:
                # The image pipeline is the guaranteed one; the text pipeline
                # is a separate, heavier optional install.
                raise ImportError(
                    "TRELLIS text-to-3D requires TrellisTextTo3DPipeline from "
                    "the `trellis` package; only the image pipeline is "
                    "available in this environment."
                ) from e
            self._text_pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
            if torch is not None and torch.cuda.is_available():
                self._text_pipeline.cuda()
        return self._text_pipeline

    def generate(self, input_kind, prompt, images, params):
        seed = self.resolve_params(params)["seed"]
        if input_kind == "text":
            if not prompt:
                raise ValueError("TRELLIS text-to-3D requires a prompt")
            pipeline = self._load_text()
            outputs = pipeline.run(prompt, seed=seed)
        else:
            if not images:
                raise ValueError("TRELLIS requires at least one image input")
            pipeline = self._load()
            outputs = pipeline.run(images[0], seed=seed) if len(images) == 1 else pipeline.run_multi_image(images, seed=seed)
        return _to_trimesh(outputs["mesh"][0])


class InstantMeshAdapter(CubeAdapter):
    """No pip-installable package exists upstream for TencentARC/InstantMesh
    today, so this always resolves to the placeholder cube. Swap in a real
    adapter here once a package lands."""
    match_tokens = ["instantmesh", "instant-mesh", "instant_mesh"]
    input_kinds = ["image", "multi_image"]

    def __init__(self, model_path):
        super().__init__(input_kinds=self.input_kinds)


class Wonder3DAdapter(CubeAdapter):
    """No pip-installable package exists upstream for Wonder3D/SV3D/Zero123
    today, so this always resolves to the placeholder cube."""
    match_tokens = ["wonder3d", "sv3d", "zero123", "zero-1-2-3"]
    input_kinds = ["image", "multi_image"]

    def __init__(self, model_path):
        super().__init__(input_kinds=self.input_kinds)


class CrmLgmAdapter(CubeAdapter):
    """No pip-installable package exists upstream for CRM/LGM today, so this
    always resolves to the placeholder cube."""
    match_tokens = ["crm", "lgm"]
    input_kinds = ["image", "multi_image"]

    def __init__(self, model_path):
        super().__init__(input_kinds=self.input_kinds)


class Hunyuan3DAdapter(MeshAdapter):
    """Adapter for tencent/Hunyuan3D-2, following its documented `hy3dgen`
    package API. Text input isn't supported by the shape-generation
    pipeline (only image/multi_image)."""
    name = "hunyuan3d"
    input_kinds = ["text", "image", "multi_image"]
    match_tokens = ["hunyuan3d", "hunyuan-3d", "hunyuan_3d"]
    PARAM_SPEC = [
        {"name": "steps", "type": "int", "default": 30, "min": 1, "max": 150},
        {"name": "guidance_scale", "type": "float", "default": 5.0, "min": 0.0, "max": 30.0},
        {"name": "octree_resolution", "type": "int", "default": 256, "min": 64, "max": 512},
    ]

    def __init__(self, model_path):
        self.model_path = model_path
        self._pipeline = None

    def is_available(self):
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return trimesh is not None
        except ImportError:
            return False

    @staticmethod
    def _has_model_weights(subfolder_path):
        """Check if a subfolder contains model weight files (.ckpt or .safetensors)."""
        if not os.path.isdir(subfolder_path):
            return False
        return any(
            name.startswith("model") and (name.endswith(".ckpt") or name.endswith(".safetensors"))
            for name in os.listdir(subfolder_path)
        )

    @staticmethod
    def _find_ckpt(dit_dir):
        """Find the best checkpoint file in a dit directory.
        Prefers .ckpt over .safetensors, fp16 over full precision."""
        candidates = [
            f for f in os.listdir(dit_dir)
            if f.startswith("model") and (f.endswith(".ckpt") or f.endswith(".safetensors"))
        ]
        if not candidates:
            return None, False
        for preferred in ("model.fp16.ckpt", "model.fp16.safetensors", "model.ckpt", "model.safetensors"):
            if preferred in candidates:
                return os.path.join(dit_dir, preferred), preferred.endswith(".safetensors")
        return os.path.join(dit_dir, candidates[0]), candidates[0].endswith(".safetensors")

    @staticmethod
    def _derive_hf_repo_id(folder_name):
        """Derive HF repo id and dit subfolder from a local folder name.
        e.g. 'tencent_Hunyuan3D-2.1' -> ('tencent/Hunyuan3D-2.1', 'hunyuan3d-dit-v2-1')"""
        match = re.search(r"Hunyuan3D-(2(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if not match:
            return None, None
        version = match.group(1)
        minor = version.split(".")[1] if "." in version else "0"
        repo_id = f"tencent/Hunyuan3D-{version}"
        dit_subfolder = f"hunyuan3d-dit-v2-{minor}"
        return repo_id, dit_subfolder

    def _resolve_shape_source(self):
        """Locate the Hunyuan3D SHAPE-generation DiT model directory.
        Returns (repo_root, dit_subfolder_name) where repo_root is always
        a local directory path (never an HF repo id)."""
        path = os.path.normpath(self.model_path) if self.model_path else ""
        parent = os.path.dirname(path)
        candidates = [d for d in dict.fromkeys([path, parent]) if d and os.path.isdir(d)]

        for repo_root in candidates:
            basename = os.path.basename(repo_root).lower()
            if "dit" in basename and self._has_model_weights(repo_root):
                return os.path.dirname(repo_root), os.path.basename(repo_root)
            for entry in sorted(os.listdir(repo_root)):
                entry_path = os.path.join(repo_root, entry)
                if "dit" in entry.lower() and os.path.isdir(entry_path) and self._has_model_weights(entry_path):
                    return repo_root, entry

        raise RuntimeError(
            f"Hunyuan3D shape-generation model not found for '{self.model_path}'. "
            f"Expected a local 'hunyuan3d-dit-v2-*' subfolder with model weight "
            f"files under the Hunyuan3D repo root."
        )

    @staticmethod
    def _ensure_config_yaml(dit_dir, repo_root):
        """Download config.yaml from HF if missing locally."""
        config_path = os.path.join(dit_dir, "config.yaml")
        if os.path.exists(config_path):
            return config_path
        repo_id, _ = Hunyuan3DAdapter._derive_hf_repo_id(os.path.basename(repo_root))
        if not repo_id:
            raise FileNotFoundError(
                f"config.yaml missing from {dit_dir} and cannot derive HF repo id "
                f"to download it. Please download config.yaml manually."
            )
        dit_name = os.path.basename(dit_dir)
        hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{dit_name}/config.yaml"
        print(f"Hunyuan3D: config.yaml missing, downloading from {hf_url}", file=sys.stderr)
        try:
            req = urllib.request.Request(hf_url, headers={"User-Agent": "dispos-ai/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(config_path, "wb") as f:
                f.write(data)
            print(f"Hunyuan3D: config.yaml saved to {config_path}", file=sys.stderr)
        except Exception as e:
            raise FileNotFoundError(
                f"config.yaml missing from {dit_dir} and download failed: {e}"
            ) from e
        return config_path

    @staticmethod
    def _needs_cpu_offload(ckpt_path):
        ckpt_bytes = os.path.getsize(ckpt_path)
        if not torch.cuda.is_available():
            return True
        vram_free = torch.cuda.mem_get_info()[0]
        return ckpt_bytes * 1.3 > vram_free

    @staticmethod
    def _load_ckpt_component(ckpt_path, component_key, use_safetensors):
        """Load a single component's state dict from the checkpoint without
        keeping the full checkpoint in memory."""
        if use_safetensors:
            import safetensors.torch
            full = safetensors.torch.load_file(ckpt_path, device="cpu")
            prefix = component_key + "."
            sd = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
            del full
        else:
            import gc
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.pop(component_key, {})
            del ckpt
            gc.collect()
        return sd

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline

        import gc
        import yaml
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen.pipelines import instantiate_from_config

        repo_root, subfolder = self._resolve_shape_source()
        dit_dir = os.path.join(repo_root, subfolder)
        config_path = self._ensure_config_yaml(dit_dir, repo_root)
        ckpt_path, use_safetensors = self._find_ckpt(dit_dir)
        if not ckpt_path:
            raise FileNotFoundError(f"No model weights found in {dit_dir}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        dtype = torch.float16
        has_cuda = torch.cuda.is_available()
        offload = self._needs_cpu_offload(ckpt_path)

        # The MoE DiT is 3B params — 12 GB in fp32. Use meta-device
        # construction + assign=True so we never materialise empty fp32
        # weights. Conditioner and VAE are small (~0.6 GB each) and can
        # be constructed normally; the VAE in particular has buffers not
        # in the checkpoint that must keep their init values.

        # When offloading, DiT + conditioner stay on CPU and get moved to
        # CUDA lazily by accelerate hooks. VAE stays on CUDA permanently
        # (only 0.66 GB) because _export() calls vae() then vae.latents2mesh()
        # in sequence — offload hooks would move it back to CPU between calls.
        dit_device = "cpu" if offload else ("cuda" if has_cuda else "cpu")
        vae_device = "cuda" if has_cuda else "cpu"

        print("Hunyuan3D: loading DiT model...", file=sys.stderr, flush=True)
        with torch.device("meta"):
            model = instantiate_from_config(config["model"])
        sd = self._load_ckpt_component(ckpt_path, "model", use_safetensors)
        sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
        model.load_state_dict(sd, strict=True, assign=True)
        del sd; gc.collect()
        model = model.to_empty(device=dit_device).to(dtype=dtype)
        gc.collect()
        if dit_device == "cuda":
            torch.cuda.empty_cache()

        print("Hunyuan3D: loading conditioner...", file=sys.stderr, flush=True)
        conditioner = instantiate_from_config(config["conditioner"])
        sd = self._load_ckpt_component(ckpt_path, "conditioner", use_safetensors)
        if sd:
            conditioner.load_state_dict(sd)
        del sd; gc.collect()
        conditioner = conditioner.to(device=dit_device, dtype=dtype)

        print("Hunyuan3D: loading VAE...", file=sys.stderr, flush=True)
        vae = instantiate_from_config(config["vae"])
        sd = self._load_ckpt_component(ckpt_path, "vae", use_safetensors)
        vae.load_state_dict(sd, strict=False)
        del sd; gc.collect()
        vae = vae.to(device=vae_device, dtype=dtype)

        image_processor = instantiate_from_config(config["image_processor"])
        scheduler = instantiate_from_config(config["scheduler"])

        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline(
            vae=vae, model=model, scheduler=scheduler,
            conditioner=conditioner, image_processor=image_processor,
            device="cuda" if has_cuda else "cpu",
            dtype=dtype,
        )

        if offload:
            print("Hunyuan3D: enabling CPU offload (DiT + conditioner)...",
                  file=sys.stderr, flush=True)
            from accelerate import cpu_offload_with_hook
            hook = None
            for part in ("conditioner", "model"):
                mod = getattr(self._pipeline, part, None)
                if isinstance(mod, torch.nn.Module):
                    _, hook = cpu_offload_with_hook(mod, "cuda", prev_module_hook=hook)
            self._pipeline.device = torch.device("cuda")

        return self._pipeline

    def generate(self, input_kind, prompt, images, params):
        if input_kind == "text":
            raise ValueError(
                "Hunyuan3D-2 shape generation does not support text input; provide an image."
            )
        if not images:
            raise ValueError("Hunyuan3D requires at least one image input")
        pipeline = self._load()

        resolved = self.resolve_params(params)
        steps = resolved["steps"]
        guidance_scale = resolved["guidance_scale"]
        octree_resolution = resolved["octree_resolution"]

        result = pipeline(
            image=images[0],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )
        return _to_trimesh(result[0])


class PointEAdapter(MeshAdapter):
    """Guarded shell for openai/point-e. Calls the official sampler + marching
    cubes utilities when the `point_e` package (installed from the GitHub
    checkout) is importable, otherwise the caller falls back to `CubeAdapter`."""
    name = "point-e"
    input_kinds = ["text", "image"]
    match_tokens = ["point-e", "point_e", "pointe"]
    PARAM_SPEC = [
        {"name": "guidance_scale", "type": "float", "default": 3.0, "min": 0.0, "max": 30.0},
    ]

    def __init__(self, model_path):
        self.model_path = model_path

    def is_available(self):
        try:
            from point_e.diffusion.configs import DIFFUSION_CONFIGS  # noqa: F401
            return trimesh is not None and torch is not None
        except ImportError:
            return False

    def _load(self, model_name):
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
        from point_e.models.download import load_checkpoint
        from point_e.models.configs import MODEL_CONFIGS, model_from_config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_from_config(MODEL_CONFIGS[model_name], device)
        model.eval()
        model.load_state_dict(load_checkpoint(model_name, device))
        diffusion = diffusion_from_config(DIFFUSION_CONFIGS[model_name])
        return model, diffusion, device

    def generate(self, input_kind, prompt, images, params):
        from point_e.diffusion.sampler import PointCloudSampler
        from point_e.util.pc_to_mesh import marching_cubes_mesh

        model_name = "base40M-textvec" if input_kind == "text" else "base40M"
        model, diffusion, device = self._load(model_name)
        sampler = PointCloudSampler(
            device=device,
            models=[model],
            diffusions=[diffusion],
            num_points=[1024],
            aux_channels=["R", "G", "B"],
            guidance_scale=[self.resolve_params(params)["guidance_scale"]],
            model_kwargs_key_filter=("texts",) if input_kind == "text" else ("*",),
        )
        model_kwargs = {"texts": [prompt or ""]} if input_kind == "text" else {}

        total = 0
        try:
            total = int(diffusion.num_timesteps)
        except (AttributeError, TypeError):
            try:
                total = len(diffusion.betas)
            except (AttributeError, TypeError):
                total = 0
        _progress_update(phase="generating", status="running", step=0, total=total)

        samples = None
        step = 0
        for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=model_kwargs):
            samples = x
            step += 1
            _progress_update(step=step, total=total)
        point_cloud = sampler.output_to_point_clouds(samples)[0]
        mesh = marching_cubes_mesh(pc=point_cloud, model=model, batch_size=4096, grid_size=32)
        return _to_trimesh(mesh)


def _create_kill_on_close_job():
    """Windows-only: create a Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    so the OS force-kills every process assigned to it as soon as the job
    handle is closed — including when this python process itself is
    TerminateProcess'd by its Rust parent, which skips atexit/close()/__del__
    and would otherwise orphan llama-server.exe holding GPU VRAM. Returns the
    job HANDLE, or None if unsupported/failed (caller falls back to the
    existing atexit/close() cleanup)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9  # JobObjectExtendedLimitInformation

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            print(f"CreateJobObjectW failed, err={ctypes.get_last_error()}", file=sys.stderr)
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        ok = kernel32.SetInformationJobObject(
            job, JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info), ctypes.sizeof(info),
        )
        if not ok:
            print(f"SetInformationJobObject failed, err={ctypes.get_last_error()}", file=sys.stderr)
            kernel32.CloseHandle(job)
            return None

        return job
    except Exception as exc:
        print(f"Failed to create kill-on-close job object: {exc}", file=sys.stderr)
        return None


def _assign_process_to_job(job, proc):
    """Windows-only: assign a Popen'd process to a job object created by
    `_create_kill_on_close_job()`. Never raises; logs and no-ops on failure
    so it can never prevent the model from running."""
    if job is None or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        proc_handle = wintypes.HANDLE(int(proc._handle))
        ok = kernel32.AssignProcessToJobObject(job, proc_handle)
        if not ok:
            print(
                f"AssignProcessToJobObject failed, err={ctypes.get_last_error()}; "
                "falling back to atexit/close() cleanup only",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"Failed to assign process to job object: {exc}", file=sys.stderr)


class LlamaMeshAdapter(MeshAdapter):
    """Adapter for LLaMA-Mesh (LLaMA-3.1-8B GGUF, text-to-mesh). Runs the GGUF
    through LM Studio's bundled `llama-server.exe` (there is no
    `llama-cpp-python` in this venv) and parses the OBJ mesh the model emits
    as plain text out of the chat completion."""
    name = "llama-mesh"
    input_kinds = ["text"]
    match_tokens = ["llama-mesh", "llama_mesh", "llamamesh"]

    # Matches nv-tlabs/LLaMA-Mesh's reference app.py inference defaults.
    LLAMA_MESH_TEMPERATURE = 0.95
    LLAMA_MESH_MAX_TOKENS = 8192

    # Verbs after which the user's prompt already reads as a direct
    # instruction (matches the model-card phrasing "Create a 3D model of a
    # chair."), so it's fed through as-is instead of double-wrapped.
    _INSTRUCTION_VERBS = ("create", "make", "generate", "design", "build", "draw", "model")

    def __init__(self, model_path):
        self.model_path = model_path
        self._proc = None
        self._port = None
        self._job = None

    def is_available(self):
        return trimesh is not None and self._find_llama_server_binary() is not None

    @staticmethod
    def _find_llama_server_binary():
        for env_var in ("USERPROFILE", "HOMEPATH", "HOME"):
            home = os.environ.get(env_var)
            if not home:
                continue
            backends_dir = os.path.join(home, ".lmstudio", "extensions", "backends")
            try:
                entries = os.listdir(backends_dir)
            except OSError:
                continue
            for entry in entries:
                candidate = os.path.join(backends_dir, entry, "llama-server.exe")
                if os.path.exists(candidate):
                    return candidate
        return None

    @staticmethod
    def _free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _ensure_server(self):
        if self._proc is not None and self._proc.poll() is None:
            return

        server_bin = self._find_llama_server_binary()
        if server_bin is None:
            raise RuntimeError(
                "LLaMA-Mesh requires LM Studio's llama-server.exe, but no "
                "backend was found under "
                "%USERPROFILE%\\.lmstudio\\extensions\\backends. Install/open "
                "LM Studio at least once so it downloads a llama.cpp backend."
            )

        self._port = self._free_port()
        print(f"Spawning llama-server for LLaMA-Mesh on port {self._port}...", file=sys.stderr)
        self._proc = subprocess.Popen(
            [
                server_bin,
                "-m", self.model_path,
                "--port", str(self._port),
                "-ngl", "99",
                "-c", "8192",
                "--host", "127.0.0.1",
            ],
        )
        atexit.register(self.close)

        self._job = _create_kill_on_close_job()
        _assign_process_to_job(self._job, self._proc)

        health_url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("llama-server exited before becoming ready")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"llama-server ready on port {self._port}", file=sys.stderr)
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"llama-server did not become ready on port {self._port} within 120s")

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._job is not None:
            if os.name == "nt":
                try:
                    import ctypes
                    ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._job)
                except Exception:
                    pass
            self._job = None

    def __del__(self):
        self.close()

    @classmethod
    def _build_prompt(cls, prompt):
        p = prompt.strip()
        if p.lower().startswith(cls._INSTRUCTION_VERBS):
            return p
        return f"Create a 3D model of {p}."

    def _chat(self, prompt):
        content = self._build_prompt(prompt)
        payload = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.LLAMA_MESH_MAX_TOKENS,
            "temperature": self.LLAMA_MESH_TEMPERATURE,
            "top_p": 0.95,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_obj(text):
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("v ", "f ", "vn ", "vt ")):
                lines.append(stripped)
        return "\n".join(lines)

    def generate(self, input_kind, prompt, images, params):
        if input_kind != "text":
            raise ValueError("LLaMA-Mesh only supports text input")
        if not prompt:
            raise ValueError("LLaMA-Mesh requires a prompt")

        self._ensure_server()
        text = self._chat(prompt)
        obj_text = self._extract_obj(text)
        if not obj_text:
            raise RuntimeError(
                f"LLaMA-Mesh did not return any OBJ geometry; model output started with: {text[:200]!r}"
            )

        mesh = trimesh.load(io.StringIO(obj_text), file_type="obj", process=False)
        mesh = _to_trimesh(mesh)
        print(
            f"LLaMA-Mesh parsed {len(mesh.vertices)} vertices / {len(mesh.faces)} faces",
            file=sys.stderr,
        )
        return mesh


# ---------------------------------------------------------------------------
# Adapter selection
# ---------------------------------------------------------------------------

# Single ordered registry of adapter classes. Order is match precedence:
# the first class whose `match_tokens` matches wins. See module docstring
# for the "add a new architecture" recipe.
ADAPTER_REGISTRY = [
    LlamaMeshAdapter,
    PointEAdapter,
    ShapEAdapter,
    TripoSRAdapter,
    SF3DAdapter,
    VggtAdapter,
    InstantMeshAdapter,
    TrellisAdapter,
    Hunyuan3DAdapter,
    Wonder3DAdapter,
    CrmLgmAdapter,
]


def _combined_name(model_path):
    normalized = os.path.normpath(model_path)
    name = os.path.basename(normalized)
    parent = os.path.basename(os.path.dirname(normalized))
    return f"{parent} {name}".lower()


def select_adapter(model_path):
    name = _combined_name(model_path)

    for adapter_cls in ADAPTER_REGISTRY:
        if not any(token in name for token in adapter_cls.match_tokens):
            continue
        adapter = adapter_cls(model_path)
        if adapter.is_available():
            print(f"Selected 3D adapter: {adapter.name}", file=sys.stderr)
            return adapter
        print(
            f"Adapter '{adapter_cls.__name__}' unavailable (missing deps); falling back to placeholder cube.",
            file=sys.stderr,
        )
        return CubeAdapter(input_kinds=adapter_cls.input_kinds)

    print("No specific 3D adapter matched model path; using placeholder cube.", file=sys.stderr)
    return CubeAdapter(input_kinds=["image"])


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status, payload):
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
        elif self.path == "/capabilities":
            self._send_json(200, {
                "input_kinds": ADAPTER.input_kinds,
                "pipeline": ADAPTER.name,
                "output_formats": ADAPTER.output_formats,
            })
        elif self.path == "/schema":
            self._send_json(200, {
                adapter_cls.name: {"params": adapter_cls.PARAM_SPEC}
                for adapter_cls in ADAPTER_REGISTRY
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            _progress_reset()

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            req = json.loads(raw.decode("utf-8")) if raw else {}

            prompt = req.get("prompt") or ""
            input_kind = req.get("input_kind") or ("text" if prompt else "image")

            if trimesh is None:
                self._send_json(500, {"error": "trimesh is not installed on the server"})
                return
            if Image is None:
                self._send_json(500, {"error": "Pillow (PIL) is not installed on the server"})
                return

            images = []
            for b64 in req.get("images") or []:
                img_bytes = base64.b64decode(b64)
                images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            if input_kind in ("image", "multi_image") and not images:
                self._send_json(400, {"error": f"input_kind '{input_kind}' requires at least one image"})
                return
            if input_kind == "text" and not prompt:
                self._send_json(400, {"error": "input_kind 'text' requires a prompt"})
                return

            _progress_update(phase="generating")
            mesh = ADAPTER.generate(input_kind, prompt, images, req)
            data, fmt = export_mesh(mesh, req.get("output_format"))
            mesh_base64 = base64.b64encode(data).decode("ascii")

            _progress_update(status="done", phase="done", step=PROGRESS.get("total", 0))

            self._send_json(200, {"mesh_base64": mesh_base64, "format": fmt})
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            _progress_update(status="error", phase="error", message=str(e))
            self._send_json(500, {"error": str(e)})


def main():
    global ADAPTER

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    print(f"Selecting 3D adapter for {args.model_path}...", file=sys.stderr)
    ADAPTER = select_adapter(args.model_path)
    print(f"3D adapter ready: {ADAPTER.name} (input_kinds={ADAPTER.input_kinds})", file=sys.stderr)

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
