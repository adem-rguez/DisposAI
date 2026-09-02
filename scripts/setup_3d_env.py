"""
Builds the dedicated Python 3.11 environment (`.venv-3d/`) that
`scripts/threed_server.py` runs under for the heavy 3D-generation
architectures (TripoSR, Hunyuan3D-2, TRELLIS, Stable-Fast-3D, VGGT). The app's
system Python is 3.14, which can't build these packages' native extensions —
this script fixes that WITHOUT requiring a system-wide Python 3.11 install or
any ComfyUI runtime: it uses `uv` to fetch a standalone CPython 3.11 and pip
(via `uv pip`) to populate it, reusing community *prebuilt wheels* only.

Safe to run under the system Python (3.14 or otherwise) — this script only
shells out to `uv`/`pip`; it does not import any of the packages it installs.

Usage:
    python scripts/setup_3d_env.py

Env vars (all optional):
    DISPOS_3D_TORCH  torch version to install (default "2.6.0")
    DISPOS_3D_CUDA   CUDA wheel tag from https://download.pytorch.org/whl/
                      (default "cu124")

Each install stage is wrapped in its own try/except so one failure (e.g. a
stale prebuilt-wheel URL, a gated HF repo, a Windows-unfriendly package)
never aborts the rest of the script. A per-stage OK/FAILED summary, plus a
final per-architecture READY/FAILED report, is printed at the end.

After running, point the daemon at this venv automatically (it's discovered
by `crates/backends/mesh-backend/src/lib.rs`'s `resolve_3d_python()` at
`.venv-3d/Scripts/python.exe` / `.venv-3d/bin/python`), or override with the
`DISPOS_3D_PYTHON` env var to use an existing Python 3.11 elsewhere.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_setup_common import ensure_uv, create_venv, uv_pip_install, stage, progress, resolve_venv_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = resolve_venv_root(PROJECT_ROOT) / ".venv-3d"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
INSTALL_MARKER = VENV_DIR / ".install_complete"

# torch >= 2.6 is required: diffusers refuses to load pickle (.bin) checkpoints
# under older torch due to CVE-2025-32434 (torch.load RCE), and several 3D
# model repos — openai/shap-e's renderer among them — ship only .bin weights.
TORCH_VERSION = os.environ.get("DISPOS_3D_TORCH", "2.6.0")
CUDA_TAG = os.environ.get("DISPOS_3D_CUDA", "cu124")
TORCH_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_TAG}"

# Comfy3D's prebuilt-wheels release house native extensions (nvdiffrast,
# spconv, torchmcubes, custom_rasterizer, diff-gaussian-rasterization,
# diffoctreerast) as manylinux/win_amd64 wheels compiled against specific
# python+torch+cuda combos. This URL is a general index into that project's
# GitHub Releases; wheel filenames (and therefore what pip resolves) are
# pinned to the exact combo they were built for and WILL drift out of sync
# with TORCH_VERSION/CUDA_TAG above over time. If installs from it start
# failing, check https://github.com/MrForExample/Comfy3D_Pre_Builds/releases
# for the current filenames and update COMFY3D_WHEELS_INDEX (or install the
# matching wheel URL directly) rather than assuming the package is
# unavailable altogether.
COMFY3D_WHEELS_INDEX = "https://github.com/MrForExample/Comfy3D_Pre_Builds/releases/expanded_assets/all_wheels"
COMFY3D_RELEASES_PAGE = "https://github.com/MrForExample/Comfy3D_Pre_Builds/releases"

STAGE_RESULTS = []  # (stage_label, ok: bool)
ARCH_RESULTS = []   # (arch_label, ok: bool)


def _print_header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Stage 1: CUDA torch
# ---------------------------------------------------------------------------

def install_torch():
    uv_pip_install(
        VENV_PYTHON,
        [f"torch=={TORCH_VERSION}", "torchvision", "torchaudio"],
        index_url=TORCH_INDEX_URL,
    )


# ---------------------------------------------------------------------------
# Stage 2: core deps
# ---------------------------------------------------------------------------

def install_core_deps():
    uv_pip_install(
        VENV_PYTHON,
        [
            "diffusers", "trimesh", "transformers", "accelerate", "rembg",
            "onnxruntime", "numpy<2", "pillow", "scikit-image", "omegaconf",
            "einops",
        ],
    )


# ---------------------------------------------------------------------------
# Stage 3: native extensions from Comfy3D prebuilt wheels (each guarded
# individually — a stale wheel URL for one package shouldn't block the rest)
# ---------------------------------------------------------------------------

def install_native_extensions():
    packages = [
        "nvdiffrast",
        "spconv",
        "torchmcubes",
        "custom_rasterizer",
        "diff-gaussian-rasterization",
        "diffoctreerast",
    ]
    for pkg in packages:
        print(f"\n  -- native extension: {pkg} --")
        try:
            uv_pip_install(VENV_PYTHON, [pkg], find_links=COMFY3D_WHEELS_INDEX)
            STAGE_RESULTS.append((f"native ext: {pkg}", True))
        except Exception as e:
            print(
                f"[FAILED] {pkg}: {e}\n"
                f"  Fallback: browse prebuilt wheels manually at {COMFY3D_RELEASES_PAGE}\n"
                f"  and find one matching torch=={TORCH_VERSION} ({CUDA_TAG}) + Python 3.11,\n"
                f"  then `uv pip install --python \"{VENV_PYTHON}\" <wheel_url>`.",
                file=sys.stderr,
            )
            STAGE_RESULTS.append((f"native ext: {pkg}", False))


# ---------------------------------------------------------------------------
# Stage 4: per-architecture pure-python packages from GitHub
# ---------------------------------------------------------------------------

def install_triposr():
    uv_pip_install(VENV_PYTHON, ["git+https://github.com/VAST-AI-Research/TripoSR"])


def install_sf3d():
    print(
        "NOTE: stable-fast-3d's HF model repo (stabilityai/stable-fast-3d) is\n"
        "GATED — you must `huggingface-cli login` and accept its license on\n"
        "huggingface.co before the model will download at runtime."
    )
    # The stable-fast-3d repo has no pyproject.toml/setup.py at its root (only
    # its texture_baker/ and uv_unwrapper/ native-extension subfolders do), so
    # a plain `pip install git+<repo>` fails outright ("does not appear to be
    # a Python project"). Its own run.py/gradio_app.py are meant to be run
    # from a checkout of the repo root with `sf3d/` on PYTHONPATH instead, so
    # mirror that: install its plain-python deps, clone the repo, and drop a
    # .pth file pointing at the checkout. texture_baker/uv_unwrapper need a
    # CUDA toolkit matching torch's build to compile and aren't needed by our
    # adapter (which bypasses UV unwrapping/texture baking) — threed_server.py
    # stubs them out at import time instead of building them here.
    uv_pip_install(
        VENV_PYTHON,
        [
            "einops", "jaxtyping", "omegaconf", "transformers",
            "open_clip_torch", "trimesh", "huggingface-hub", "rembg",
            "pynanoinstantmeshes", "gpytoolbox",
        ],
    )

    src_dir = VENV_DIR / "_sf3d_src"
    if not (src_dir / "sf3d").is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Stability-AI/stable-fast-3d", str(src_dir)],
            check=True,
        )

    site_packages = subprocess.run(
        [str(VENV_PYTHON), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    Path(site_packages, "stable_fast_3d.pth").write_text(str(src_dir) + "\n")


def install_hunyuan3d():
    uv_pip_install(VENV_PYTHON, ["git+https://github.com/Tencent/Hunyuan3D-2"])


def install_trellis():
    print(
        "NOTE: TRELLIS (microsoft/TRELLIS) is fragile on Windows even with\n"
        "prebuilt wheels for its native deps (spconv/nvdiffrast/diffoctreerast) —\n"
        "expect this stage to be the most likely to fail here."
    )
    uv_pip_install(VENV_PYTHON, ["git+https://github.com/microsoft/TRELLIS"])


def install_vggt():
    # facebook/vggt-1b's `vggt` package isn't published on PyPI; install from
    # source. `open3d` provides the Poisson surface reconstruction used to
    # turn VGGT's predicted point maps into a mesh.
    uv_pip_install(VENV_PYTHON, ["git+https://github.com/facebookresearch/vggt.git"])
    uv_pip_install(VENV_PYTHON, ["open3d"])


# ---------------------------------------------------------------------------
# Final import-check summary
# ---------------------------------------------------------------------------

def _import_check(module_expr):
    proc = subprocess.run(
        [str(VENV_PYTHON), "-c", module_expr],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def check_architectures():
    _print_header("Import-checking each architecture in the venv")
    checks = [
        ("TripoSR", "from tsr.system import TSR"),
        ("Stable-Fast-3D (SF3D)", "from sf3d.system import SF3D"),
        ("Hunyuan3D-2", "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline"),
        ("TRELLIS", "from trellis.pipelines import TrellisImageTo3DPipeline"),
        ("VGGT", "from vggt.models.vggt import VGGT; import open3d"),
    ]
    for label, expr in checks:
        ok = _import_check(expr)
        ARCH_RESULTS.append((label, ok))
        print(f"  {'READY ' if ok else 'FAILED'} - {label}")


def print_summary():
    _print_header("Stage summary")
    for label, ok in STAGE_RESULTS:
        print(f"  {'OK    ' if ok else 'FAILED'} - {label}")

    _print_header("Architecture summary")
    for label, ok in ARCH_RESULTS:
        print(f"  {'READY ' if ok else 'FAILED'} - {label}")

    print(
        f"\nVenv python: {VENV_PYTHON}\n"
        "The daemon auto-discovers this venv; no further configuration needed\n"
        "unless you relocate it, in which case set DISPOS_3D_PYTHON."
    )


def _run_stage(label, fn):
    progress(label, "running")
    ok = stage(label, STAGE_RESULTS)(fn)
    progress(label, "done" if ok else "error", message=None if ok else f"{label} failed")
    return ok


def main():
    ensure_uv()

    try:
        create_venv(VENV_DIR, python_version="3.11")
    except Exception as e:
        print(f"[FAILED] Could not create {VENV_DIR}: {e}", file=sys.stderr)
        sys.exit(1)

    core_ok = True
    core_ok &= _run_stage("torch (CUDA)", install_torch)
    core_ok &= _run_stage("core deps", install_core_deps)

    _run_stage("native extensions (Comfy3D prebuilt wheels)", install_native_extensions)
    _run_stage("TripoSR", install_triposr)
    _run_stage("Stable-Fast-3D (SF3D)", install_sf3d)
    _run_stage("Hunyuan3D-2", install_hunyuan3d)
    _run_stage("TRELLIS", install_trellis)
    _run_stage("VGGT", install_vggt)

    if core_ok:
        INSTALL_MARKER.write_text("ok")

    check_architectures()
    print_summary()


if __name__ == "__main__":
    main()
