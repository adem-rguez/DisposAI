"""
Generic (transformers) text-to-speech HTTP server.

Spawned as a subprocess by the Rust `tts-backend` crate for any non-Kokoro
TTS model. Loads a `transformers.pipeline("text-to-speech", ...)` once at
startup and serves synthesis requests over a stdlib HTTP server.

This is intentionally ONE generic code path with no per-architecture
branching: `transformers.pipeline` auto-detects the right model class from
the repo, and the forward-params actually passed to it are filtered through
`inspect.signature()` at request time, mirroring
`video_diffusers_server.py`'s `_build_call_kwargs` pattern. Kokoro (ONNX,
custom G2P/voice-embedding pipeline) is handled separately by
`kokoro_tts_server.py` — this script does not attempt to generalize it.

Usage:
    python tts_server.py --model-path <path/to/model_dir_or_repo> --port <port>

Protocol:
    POST /synthesize
        body: {"text": "...", "voice": "af_heart", "speed": 1.0}
        200:  {"audio_b64": "<base64 wav>", "sample_rate": 24000, "duration_s": 3.9}
        4xx/5xx: {"error": "..."}
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

import numpy as np

# Import torch up front: creating a CUDA session with a different DLL-loading
# order than torch expects can fail with WinError 127. Loading torch first
# and injecting the nvidia package DLL directories wins. Mirrors
# kokoro_tts_server.py / video_diffusers_server.py's import ordering.
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
    import soundfile as sf
except ImportError as e:
    print(f"Failed to import soundfile: {e}", file=sys.stderr)
    sf = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError as e:
    print(f"Failed to import transformers: {e}", file=sys.stderr)
    sys.exit(1)

PIPE = None
DEVICE = None
SPEAKER_EMBEDDING = None


def _load_default_speaker_embedding():
    """Best-effort default speaker embedding for architectures (e.g. SpeechT5)
    that require one via forward_params. If this fails for any reason, return
    None and let the pipeline raise its own clear error instead."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        return torch.tensor(dataset[7305]["xvector"]).unsqueeze(0)
    except Exception as e:
        print(f"Could not load default speaker embedding: {e}", file=sys.stderr)
        return None


def _needs_speaker_embedding() -> bool:
    if PIPE is None:
        return False
    model = getattr(PIPE, "model", None)
    class_name = type(model).__name__ if model is not None else ""
    return "SpeechT5" in class_name


def _build_forward_params(req: dict) -> dict:
    """Filter request fields through whatever forward-params the loaded
    pipeline's underlying model/generate signature actually accepts, so only
    kwargs the specific architecture supports are ever passed. Most HF TTS
    pipelines accept none of these, in which case this returns {}."""
    model = getattr(PIPE, "model", None)
    accepted = set()
    for attr in ("forward", "generate"):
        fn = getattr(model, attr, None)
        if fn is not None:
            try:
                accepted |= set(inspect.signature(fn).parameters.keys())
            except (TypeError, ValueError):
                pass

    forward_params = {}

    speed = req.get("speed")
    if speed is not None:
        for candidate_name in ("speed", "speaking_rate"):
            if candidate_name in accepted:
                forward_params[candidate_name] = float(speed)
                break

    voice = req.get("voice")
    if voice is not None and "speaker_id" in accepted:
        forward_params["speaker_id"] = voice

    if SPEAKER_EMBEDDING is not None and "speaker_embeddings" in accepted:
        forward_params["speaker_embeddings"] = SPEAKER_EMBEDDING

    return forward_params


def synthesize(text: str, forward_params: dict):
    if forward_params:
        result = PIPE(text, forward_params=forward_params)
    else:
        result = PIPE(text)

    audio = result["audio"]
    sample_rate = result["sampling_rate"]

    audio = np.asarray(audio).squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    wav_bytes = buf.getvalue()

    duration_s = len(audio) / sample_rate
    return wav_bytes, sample_rate, duration_s


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

    def do_POST(self):
        if self.path != "/synthesize":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))

            text = req.get("text", "")
            if not text:
                self._send_json(400, {"error": "'text' is required"})
                return

            forward_params = _build_forward_params(req)
            wav_bytes, sample_rate, duration_s = synthesize(text, forward_params)
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

            self._send_json(200, {
                "audio_b64": audio_b64,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
            })
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})


def main():
    global PIPE, DEVICE, SPEAKER_EMBEDDING

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    if sf is None:
        print("soundfile is required to encode WAV output", file=sys.stderr)
        sys.exit(1)

    print(f"Loading transformers TTS pipeline from {args.model_path}...", file=sys.stderr)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device_arg = 0 if DEVICE == "cuda" else -1
    PIPE = hf_pipeline("text-to-speech", model=args.model_path, device=device_arg)

    print(f"transformers TTS pipeline loaded on device: {DEVICE}", file=sys.stderr)

    if _needs_speaker_embedding():
        SPEAKER_EMBEDDING = _load_default_speaker_embedding()

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
