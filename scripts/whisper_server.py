"""
HF Transformers ASR (speech-to-text) HTTP server.

Spawned as a subprocess by the Rust `whisper-backend` crate. Loads any
HuggingFace Whisper-family checkpoint once at startup via
`transformers.pipeline("automatic-speech-recognition", ...)` and serves
transcription requests over a stdlib HTTP server. Mirrors the
spawn-a-python-server pattern used by `kokoro_tts_server.py`,
`diffusers_server.py`, `video_diffusers_server.py` and
`hf_transformers_server.py`.

Usage:
    python whisper_server.py --model-path <path/to/model_dir_or_file> --port <port>

Protocol:
    POST /transcribe
        body: {"audio_b64": "<base64 audio bytes>", "language": "en" (optional)}
        200:  {"text": "..."}
        4xx/5xx: {"error": "..."}
    GET /health -> {"status": "ok"}

Audio decoding uses `soundfile` (libsndfile), which natively supports WAV,
FLAC, OGG and MP3. No ffmpeg dependency (none of the other *_server.py
scripts in this repo assume ffmpeg is on PATH either). Non-16kHz audio is
resampled with `scipy.signal.resample_poly`, and stereo is downmixed to mono,
since Whisper's feature extractor expects 16kHz mono.

Prints "READY" to stdout once the model is loaded and the server is
listening. The Rust backend watches for this line.
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np

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
    import soundfile as sf
except ImportError as e:
    print(f"Failed to import soundfile: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import pipeline
except ImportError as e:
    print(f"Failed to import transformers/torch: {e}", file=sys.stderr)
    sys.exit(1)

TARGET_SR = 16000
ASR_PIPE = None


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes (wav/flac/ogg/mp3) to a mono float32
    waveform at TARGET_SR, as expected by Whisper's feature extractor."""
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    # Downmix to mono.
    waveform = data.mean(axis=1)

    if sample_rate != TARGET_SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sample_rate, TARGET_SR)
        up, down = TARGET_SR // g, sample_rate // g
        waveform = resample_poly(waveform, up, down).astype(np.float32)

    return waveform


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
        if self.path != "/transcribe":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))

            audio_b64 = req.get("audio_b64")
            if not audio_b64:
                self._send_json(400, {"error": "'audio_b64' is required"})
                return

            audio_bytes = base64.b64decode(audio_b64)
            waveform = decode_audio(audio_bytes)

            generate_kwargs = {}
            language = req.get("language")
            if language:
                generate_kwargs["language"] = language

            result = ASR_PIPE(
                {"raw": waveform, "sampling_rate": TARGET_SR},
                generate_kwargs=generate_kwargs or None,
            )
            text = result.get("text", "").strip() if isinstance(result, dict) else str(result)

            self._send_json(200, {"text": text})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})


def main():
    global ASR_PIPE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading Whisper ASR model from {args.model_path}...", file=sys.stderr)

    device = 0 if torch.cuda.is_available() else -1
    ASR_PIPE = pipeline(
        "automatic-speech-recognition",
        model=args.model_path,
        torch_dtype="auto",
        device=device,
    )
    print(f"ASR pipeline loaded on device={device}", file=sys.stderr)

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
