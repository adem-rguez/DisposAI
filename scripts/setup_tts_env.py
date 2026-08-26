"""
Builds the dedicated Python 3.11 environment (`.venv-tts/`) that
`scripts/kokoro_tts_server.py` and `scripts/tts_server.py` run under for
text-to-speech synthesis. Uses `uv` to fetch a standalone CPython 3.11 and
pip (via `uv pip`) to populate it — no system-wide Python 3.11 install
required.

Safe to run under the system Python — this script only shells out to
`uv`/`pip`; it does not import any of the packages it installs.

Usage:
    python scripts/setup_tts_env.py

Env vars (all optional):
    DISPOS_TTS_TORCH  torch version to install (default "2.6.0")
    DISPOS_TTS_CUDA   CUDA wheel tag from https://download.pytorch.org/whl/
                       (default "cu124")
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_setup_common import ensure_uv, create_venv, uv_pip_install, stage, progress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv-tts"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
INSTALL_MARKER = VENV_DIR / ".install_complete"

TORCH_VERSION = os.environ.get("DISPOS_TTS_TORCH", "2.6.0")
CUDA_TAG = os.environ.get("DISPOS_TTS_CUDA", "cu124")
TORCH_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_TAG}"

STAGE_RESULTS = []  # (stage_label, ok: bool)


def _print_header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def install_torch():
    uv_pip_install(
        VENV_PYTHON,
        [f"torch=={TORCH_VERSION}", "torchaudio"],
        index_url=TORCH_INDEX_URL,
    )


def install_core_deps():
    # kokoro_tts_server.py: numpy, onnxruntime (CUDA execution provider),
    # huggingface_hub, misaki.espeak (G2P). tts_server.py: numpy,
    # transformers (generic transformers.pipeline("text-to-speech", ...)).
    uv_pip_install(
        VENV_PYTHON,
        [
            "numpy<2", "onnxruntime-gpu", "huggingface_hub", "misaki[en]",
            "transformers", "accelerate", "safetensors",
        ],
    )


def _run_stage(label, fn):
    progress(label, "running")
    ok = stage(label, STAGE_RESULTS)(fn)
    progress(label, "done" if ok else "error", message=None if ok else f"{label} failed")
    return ok


def print_summary():
    _print_header("Stage summary")
    for label, ok in STAGE_RESULTS:
        print(f"  {'OK    ' if ok else 'FAILED'} - {label}")
    print(
        f"\nVenv python: {VENV_PYTHON}\n"
        "The daemon auto-discovers this venv for the TTS backend "
        "(Kokoro ONNX + generic transformers TTS).\n"
        "Note: eSpeak NG must also be installed system-wide "
        "(C:\\Program Files\\eSpeak NG) for Kokoro's G2P to work."
    )


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

    if core_ok:
        INSTALL_MARKER.write_text("ok")

    print_summary()


if __name__ == "__main__":
    main()
