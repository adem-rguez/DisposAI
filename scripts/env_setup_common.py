"""
Shared helpers for the per-backend Python environment setup scripts
(`setup_3d_env.py`, `setup_sd_env.py`, `setup_video_env.py`,
`setup_tts_env.py`).

Provides `uv`-based venv creation/install helpers, a `stage()` wrapper for
recording per-stage OK/FAILED results without aborting the whole script, and
a `progress()` helper that emits a single-line `PROGRESS: {...}` JSON record
to stdout so a parent process (the Rust daemon) can stream install progress
by reading the subprocess's stdout line-by-line.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _print_header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def resolve_venv_root(project_root: Path) -> Path:
    """Return the root directory for venv folders. If DISPOS_VENV_ROOT env var
    is set and non-empty, use that; otherwise use project_root."""
    venv_root = os.environ.get("DISPOS_VENV_ROOT", "").strip()
    if venv_root:
        return Path(venv_root)
    return project_root


def _run(cmd, cwd=None):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"command exited with status {proc.returncode}")


def progress(phase: str, status: str, message: str = None):
    """Emit a machine-readable progress line. `status` is one of
    "running"/"done"/"error". Parent processes grep stdout for the
    `PROGRESS: ` prefix; everything else on stdout is just human-readable
    log output."""
    record = {"phase": phase, "status": status, "message": message}
    print(f"PROGRESS: {json.dumps(record)}", flush=True)


def ensure_uv():
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path
    _print_header("`uv` not found")
    print(
        "This script needs `uv` (https://github.com/astral-sh/uv) to fetch a\n"
        "standalone Python with no system install required.\n\n"
        "Install it with:\n"
        "    pip install uv\n\n"
        "...then re-run this script."
    )
    sys.exit(1)


def create_venv(venv_dir: Path, python_version: str = "3.11"):
    _print_header(f"Creating {venv_dir} (Python {python_version})")
    if venv_dir.exists():
        for attempt in range(3):
            try:
                shutil.rmtree(venv_dir)
                break
            except OSError:
                if attempt == 2:
                    raise RuntimeError(
                        f"Could not remove existing directory {venv_dir}: a process "
                        "is likely still using it (e.g. a leftover backend python.exe "
                        "process). Close any running DisposAI backend processes, or "
                        "restart the app, and try again."
                    )
                time.sleep(1)
    _run(["uv", "venv", str(venv_dir), "--python", python_version])
    venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not venv_python.exists():
        raise RuntimeError(f"venv python not found at expected path: {venv_python}")
    return venv_python


def uv_pip_install(venv_python: Path, packages, index_url=None, find_links=None):
    cmd = ["uv", "pip", "install", "--python", str(venv_python)]
    if index_url:
        cmd += ["--index-url", index_url]
    if find_links:
        cmd += ["--find-links", find_links]
    cmd += list(packages)
    _run(cmd)


def stage(label, results):
    """Decorator-less helper: run `fn()`, record OK/FAILED into `results`
    (a list of `(label, ok)` tuples), never raise."""
    def _wrap(fn):
        print(f"\n--- {label} ---")
        try:
            fn()
            results.append((label, True))
            return True
        except Exception as e:
            print(f"[FAILED] {label}: {e}", file=sys.stderr)
            results.append((label, False))
            return False
    return _wrap
