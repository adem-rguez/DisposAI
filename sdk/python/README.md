# DisposAI Python SDK

Official Python client for the **DisposAI** local inference daemon.

## Installation

```bash
pip install dispos-sdk
```

Or install in development mode from the repository:

```bash
cd sdk/python
pip install -e .
```

## Quick Start

```python
from dispos_sdk import DisposClient

client = DisposClient("http://localhost:8080")

# 1. Check if model fits in VRAM/RAM before loading
fit = client.estimate_fit(
    parameter_count_billions=7.0,
    quantization="Q4_K_M",
    context_size=4096
)
print(f"Fits in VRAM: {fit['fits_in_vram']} (Est. {fit['estimated_tok_per_sec']} tok/s)")

# 2. OpenAI-compatible chat completion
response = client.chat_completion(
    model="models/Qwen3.5-0.8B-Q8_0.gguf",
    messages=[{"role": "user", "content": "Write a fast Rust binary search function."}]
)
print(response["choices"][0]["message"]["content"])
```

## License

MIT OR Apache-2.0
