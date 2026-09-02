<div align="center">

# ⚡ DisposAI

**A high-performance, resident local inference daemon & multimodal AI studio written in Rust.**  
*Every model at your disposal — powered by Dynamic Model Toolization (DMT), Zero-OOM memory budgeting, native CUDA acceleration, and drop-in OpenAI API compatibility.*

[![CI](https://github.com/adem-rguez/DisposAI/actions/workflows/ci.yml/badge.svg)](https://github.com/adem-rguez/DisposAI/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.78%2B-orange?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Tokio](https://img.shields.io/badge/Async-Tokio%20%7C%20Axum-blue?style=for-the-badge&logo=tokio&logoColor=white)](https://tokio.rs/)
[![CUDA](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Electron](https://img.shields.io/badge/Desktop-Electron%20%7C%20React%2019-47848F?style=for-the-badge&logo=electron&logoColor=white)](https://electronjs.org/)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-blue.svg?style=for-the-badge)](#license)

[Dynamic Model Toolization (DMT)](#-dynamic-model-toolization-dmt) • [Key Features](#-key-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Studios](#-multimodal-studios) • [API & SDKs](#-api--sdks)

</div>

---

## 🎬 Demo

<div align="center">

![DisposAI Demo](demos/demo%2002-09-2026.webp)

<sub>(Dispos Studio walkthrough — chat, image generation, and 3D mesh generation via Dynamic Model Toolization)</sub>

</div>

---

## 🚀 Why DisposAI?

The name **Dispos** comes from the Latin word *dispositor* (from *disponere*), which means *"arranger"* or *"one who sets things in order"*—reflecting how the system orchestrates and manages your suite of local models.

Running local AI today is fragmented and prone to out-of-memory crashes (**OOM**). Users juggle multiple heavy tools—one for LLMs, one for Stable Diffusion, another for 3D generation, and separate scripts for TTS—often crashing when VRAM allocations collide.

DisposAI unifies text reasoning, vision understanding, 3D mesh generation, image synthesis, voice synthesis, and transcription under **one protocol, one memory arbiter, one dashboard, and one desktop studio**.

---

## 🔮 Dynamic Model Toolization (DMT)

> **"Subagents for Local Multi-Model Intelligence"**

In modern agentic workflows (such as Claude Code subagents), an orchestrator accomplishes complex tasks by delegating to specialized subagents. **DisposAI brings this paradigm directly to local consumer hardware.**

Through **Dynamic Model Toolization (DMT)**, the orchestrator LLM is not confined to its own weights. Instead, it treats **every other model installed on your machine as a dynamic, invocable tool**:

```
                                  ┌───────────────────────────┐
                                  │      User Prompt          │
                                  │ "Generate a 3D rover and  │
                                  │  narrate its mission."    │
                                  └─────────────┬─────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │  ORCHESTRATOR LLM (GGUF)  │
                                  │   (Autonomous Reasoning)  │
                                  └──────┬─────────────┬──────┘
                                         │             │
                    1. list_models()     │             │ 2. run_model("sd-v1-5")
                    Discovers inventory  │             │ Auto-loads SD & renders image
                                         ▼             ▼
                                  ┌───────────────────────────┐
                                  │  Stable Diffusion Backend │
                                  │  -> Returns Image Handle  │
                                  └─────────────┬─────────────┘
                                                │
                                                │ 3. run_model("stable-fast-3d", image=handle)
                                                │ Auto-loads SF3D & generates 3D Mesh
                                                ▼
                                  ┌───────────────────────────┐
                                  │    Mesh3D Backend         │
                                  │    -> Returns 3D GLB Mesh │
                                  └─────────────┬─────────────┘
                                                │
                                                │ 4. run_model("kokoro-tts", text=summary)
                                                │ Auto-loads Kokoro & generates Audio
                                                ▼
                                  ┌───────────────────────────┐
                                  │    Kokoro TTS Backend     │
                                  │    -> Returns Audio WAV   │
                                  └─────────────┬─────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │     SYNTHESIZED RESULT    │
                                  │ Interactive 3D Viewer +   │
                                  │ Audio Playback + Markdown │
                                  └───────────────────────────┘
```

### How DMT Works:
1. 🔍 **Dynamic Discovery (`list_models`)**: The orchestrator inspects the machine's local catalog to check available modalities (Text, Vision, Image, 3D Mesh, Voice, Video) and whether they are currently loaded.
2. ⚡ **On-Demand Auto-Loading (`run_model`)**: When a toolized model is called, the daemon automatically loads it onto a dynamically allocated port without requiring manual user intervention.
3. 🔗 **Multimodal Artifact Chaining**: Media outputs (image handles, 3D meshes, audio streams) generated by one model are seamlessly passed as inputs to subsequent models (e.g. text prompt $\rightarrow$ Stable Diffusion $\rightarrow$ TripoSR/SF3D 3D GLB mesh $\rightarrow$ Kokoro voice narration).
4. 🔄 **Multi-Hop Agentic Loop**: Supports multi-step reasoning hops per turn, allowing the LLM to verify results, retry, or chain multiple tools before delivering the final response.
5. 🪄 **Autopilot Mode**: A 1-click toggle in the desktop UI and API that gives the orchestrator full autonomy to select, load, and execute the best companion model tools automatically.

---

## ✨ Key Features

### 🛡️ 1. Zero-OOM Hardware Profiler & Pre-Download Fit Estimator
- **Active Hardware Telemetry**: Automatically probes CPU cores, system RAM, NVIDIA GPU VRAM (via NVML / `nvidia-smi`), and memory headroom.
- **Pre-Download Mathematical Fit Estimator**: Computes model weights, quantization overhead, and KV-cache sizing for your target context length to predict whether a model will fit in VRAM or RAM **before you spend time and bandwidth downloading it**.
- **VRAM Arbiter**: Prevents out-of-memory panics by budgeting allocations across concurrently active backends.

### 🧩 2. Pluggable Multimodal Backend Engine
Every modality implements a unified `InferenceBackend` Rust trait with standardized lifecycle controls (`estimate_vram`, `load_model`, `unload_model`, `generate`, `generate_stream`, `poll_progress`):

| Modality | Backend Engine | Supported Formats / Architectures |
|---|---|---|
| **💬 Text & Reasoning** | `llama.cpp` (Native CUDA) | GGUF (Qwen 2.5/3, DeepSeek-R1, Llama 3, Mistral, Phi-4) with native Jinja `<think>` reasoning extraction |
| **👁️ Multimodal Vision** | `llama.cpp` Multimodal | Vision GGUFs + auto-detected sibling `--mmproj` projectors |
| **🧊 3D Mesh Generation** | `threed_server` Adapter Engine | TripoSR, Stable-Fast-3D (SF3D), Shap-E, TRELLIS, Hunyuan3D-2, Point-E (GLB, OBJ, STL, PLY, FBX) |
| **🎨 Image Synthesis** | `stable-diffusion.cpp` / Diffusers | GGUF Component Mode (auto-wiring diffusion model, CLIP-L, CLIP-G, T5XXL, VAE) & Safetensors |
| **🎙️ Voice Synthesis (TTS)** | Kokoro-82M ONNX Runtime | High-speed, natural neural speech synthesis with speed & voice controls |
| **👂 Speech Recognition (ASR)** | `whisper.cpp` | Offline automatic speech recognition and audio transcription |
| **🎬 Video Generation** | Wan Video Runner | Local text-to-video diffusion pipeline |

### 🌐 3. Integrated Hugging Face Hub Explorer
- **In-App Search & Granular Filters**: Search millions of models filtered by modality (Text, Vision, 3D, Audio, Image), parameter count ranges (e.g. 0.5B – 70B), quantizations (`Q4_K_M`, `Q8_0`, `F16`), and languages.
- **Smart Auto-Quant Sizing**: Automatically picks the best quantization variant matching your GPU's available free VRAM.
- **Gated Model Support**: Built-in Hugging Face token manager for gated repositories (Llama 3, etc.).
- **High-Speed Chunked Downloader**: Resumable multi-threaded downloads with live progress and cancellation support.

### 🖥️ 4. Futuristic Desktop Studio (Electron + React 19 + Three.js)
- **Model Center**: Manage local GGUF models, configure GPU offload layers (`-ngl`), context length, sampling parameters, and launch dedicated studios.
- **Chat Studio**: Markdown rendering with GFM support, collapsible thinking/reasoning blocks (`<think>`), image/mesh attachment dropzones, speed telemetry (tokens/sec), and Autopilot mode.
- **Interactive 3D Studio**: Real-time WebGL Three.js 3D viewport with orbit controls, wireframe toggles, camera framing, and direct `.glb` / `.obj` / `.stl` downloads.
- **Image & Voice Studios**: Parameterized generation panels with live step-by-step progress tracking.

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLIENTS & FRONTENDS                                │
│    Dispos Studio (Electron/React)   │  Web Dashboard  │   Python / TS SDKs  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP / SSE (:8080) & gRPC (:50051)
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                         DISPOS DAEMON CORE (Rust)                           │
│  - Session Manager             - System Hardware Profiler (NVML / RAM)      │
│  - VRAM Arbiter                - Pre-Download Fit Estimator                 │
│  - Dynamic Model Toolization   - Hugging Face Download & Sync Manager       │
│  - Multi-Model Registry        - Media Store & Retention Engine             │
└──────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────┘
       │              │              │              │              │
┌──────▼──────┐┌──────▼──────┐┌──────▼──────┐┌──────▼──────┐┌──────▼──────┐
│  llama.cpp  ││  SD Backend ││ Mesh Backend││ Kokoro TTS  ││ Whisper.cpp │
│  (Text/LLM, ││  (SD.cpp &  ││ (TripoSR,   ││ (ONNX Voice ││    (ASR     │
│   Vision)   ││  Diffusers) ││  SF3D, GLB) ││  Synthesis) ││Transcribing)│
└─────────────┘└─────────────┘└─────────────┘└─────────────┘└─────────────┘
```

---

## 📦 Repository Structure

```
DisposAI/
├── crates/
│   ├── daemon-core/           # Resident Rust daemon, HTTP Axum router, gRPC, profiler & arbiter
│   ├── backend-trait/         # Common InferenceBackend trait and modality definitions
│   ├── backends/
│   │   ├── llama-backend/     # llama.cpp bindings (text, reasoning, vision mmproj)
│   │   ├── sd-backend/        # stable-diffusion.cpp & diffusers image synthesis
│   │   ├── mesh-backend/      # 3D mesh generation (TripoSR, SF3D, Shap-E, GLB)
│   │   ├── tts-backend/       # Kokoro ONNX speech synthesis backend
│   │   ├── whisper-backend/   # whisper.cpp ASR transcription backend
│   │   └── video-backend/     # Wan local video generation backend
│   ├── moe-cache/             # MoE expert predictive activation cache
│   ├── spec-decode/           # Speculative decoding draft manager
│   ├── pool-protocol/         # LAN cluster device pooling protocol
│   ├── proto/                 # gRPC protobuf definitions
│   └── client-sdk/            # Rust client SDK
├── apps/
│   └── dispos-studio/         # Dispos Studio (Electron + React 19 desktop app & 3D viewer)
├── sdk/
│   ├── python/                # Python client SDK (`dispos-sdk`)
│   └── typescript/            # TypeScript client SDK
├── scripts/                   # Specialized Python inference bridges (diffusers, kokoro, 3d)
├── examples/                  # Python & API validation test scripts
└── models/                    # Local GGUF/checkpoint model storage
```

---

## ⚡ Quick Start

### Prerequisites
- **Rust** (1.78+ recommended): [Install Rust](https://www.rust-lang.org/tools/install)
- **Node.js** (v18+ & npm): [Install Node.js](https://nodejs.org/)
- **NVIDIA GPU** with CUDA drivers (for native hardware acceleration)
- *(Optional)* Python 3.10+ (for extended 3D mesh generation adapters)

### Build and Run

```bash
# Clone the repository
git clone https://github.com/adem-rguez/DisposAI.git
cd DisposAI

# Install dependencies and launch
npm install
npm start
```
*`npm start` builds the Rust daemon and launches Dispos Studio (Electron + React) in one step — no separate `cargo run` required. The daemon listens on `http://0.0.0.0:8080` (HTTP REST & Web Dashboard) and `0.0.0.0:50051` (gRPC).*

---

## 🎮 Multimodal Studios

### 💬 Chat Studio & Reasoning Engine
- **Jinja Reasoning Support**: Automatically detects and separates `<think>` reasoning traces into collapsible accordion blocks with word counts.
- **Multimodal Vision**: Drop images into the prompt area to analyze diagrams, inspect scenes, and ask visual questions.
- **Autopilot Mode**: Toggle Autopilot to let the model invoke Dynamic Model Toolization (DMT) sub-models autonomously.

### 🧊 3D Mesh Studio & Interactive Viewport
- **Text / Image / Multi-Image to 3D**: Generate meshes from text descriptions or reference images using TripoSR, Stable-Fast-3D, or Shap-E.
- **Three.js WebGL Viewport**: Rotate, pan, zoom, toggle wireframe view, inspect vertex geometry, and export `.glb`, `.obj`, `.stl`, or `.ply` files.

### 🎨 Stable Diffusion Image Studio
- Full control over prompt, negative prompt, denoising steps, CFG scale, seed, and resolution (512×512, 768×768, 1024×1024).
- Real-time step progress bar during diffusion sampling.

### 🎙️ Voice & Audio Studio
- **Kokoro-82M TTS**: High-speed neural speech synthesis with instant audio playback.
- **Whisper ASR**: Drop audio files to transcribe spoken voice to text.

---

## 🔌 API & SDKs

DisposAI serves standard OpenAI-compatible endpoints alongside specialized daemon management APIs.

### 1. OpenAI-Compatible Chat Completion (cURL)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-0.8B-Q8_0.gguf",
    "messages": [
      {"role": "user", "content": "Explain quantum superposition in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### 2. Pre-Download Fit Estimator (cURL)

```bash
curl -X POST http://localhost:8080/v1/fit-estimator \
  -H "Content-Type: application/json" \
  -d '{
    "parameter_count_billions": 14.0,
    "quantization": "Q4_K_M",
    "context_size": 8192,
    "modality": "Text"
  }'
```

### 3. Python SDK Usage

```python
from dispos_sdk import DisposClient

client = DisposClient("http://localhost:8080")

# 1. Check system fit before loading
fit = client.estimate_fit(
    parameter_count_billions=7.0,
    quantization="Q4_K_M",
    context_size=4096
)
print(f"Fits in VRAM: {fit['fits_in_vram']} (Est. {fit['estimated_tok_per_sec']} tok/s)")

# 2. Chat completion
response = client.chat_completion(
    model="models/Qwen3.5-0.8B-Q8_0.gguf",
    messages=[{"role": "user", "content": "Write a fast Rust binary search function."}]
)
print(response["choices"][0]["message"]["content"])
```

### 4. TypeScript SDK Usage

```typescript
import { DisposClient } from 'dispos-sdk';

const client = new DisposClient('http://localhost:8080');

async function main() {
  const fit = await client.estimateFit({
    parameter_count_billions: 8.0,
    quantization: 'Q8_0',
    context_size: 4096,
  });
  console.log(`Fits in GPU: ${fit.fits_in_vram}`);

  const completion = await client.chatCompletion({
    model: 'models/Qwen3.5-0.8B-Q8_0.gguf',
    messages: [{ role: 'user', content: 'Hello DisposAI!' }],
  });
  console.log(completion.choices[0].message.content);
}

main();
```

---

## 🛠️ Configuration & Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DISPOS_MODELS_DIR` | Directory where local model files (`.gguf`, `.safetensors`, `.onnx`) are stored | `./models` |
| `DISPOS_SD_BINARY` | Explicit path to `sd.exe` / `sd` binary for stable-diffusion.cpp | Auto-detected in PATH / LMStudio dir |
| `DISPOS_3D_PYTHON` | Custom Python interpreter path for the 3D mesh generation environment | `.venv-3d/Scripts/python.exe` |
| `DISPOS_DAEMON_PATH` | Path to compiled `daemon-core` binary used by the desktop launcher | `target/release/daemon-core.exe` |
| `HF_TOKEN` | Optional Hugging Face access token for gated repositories | None (configurable in UI) |

---

## 🗺️ Roadmap

- [x] High-performance resident Rust daemon with Axum HTTP and Tonic gRPC
- [x] Dynamic Model Toolization (DMT) orchestrator & multi-hop tool execution loop
- [x] Multi-model dynamic port allocation & process lifecycle tracking
- [x] Hardware profiler & mathematical pre-download VRAM/RAM fit estimator
- [x] Native CUDA LLM backend with reasoning `<think>` extraction
- [x] Vision-language input with automatic companion mmproj detection
- [x] Stable Diffusion GGUF component mode & Safetensors HTTP engine
- [x] Text/Image-to-3D Mesh generation with interactive WebGL Three.js studio
- [x] Kokoro-82M ONNX local Text-to-Speech synthesis
- [x] Integrated Hugging Face Hub search, smart quantizer recommendation, and download manager
- [x] Python and TypeScript Client SDKs
- [x] Whisper.cpp offline speech recognition (ASR)
- [x] Local text-to-video diffusion pipeline
- [ ] Embeddings modality & API route
- [ ] MoE dynamic predictive expert cache CUDA stream prefetching
- [ ] Speculative decoding automatic draft model pairing
- [ ] Multi-device LAN heterogeneous VRAM pooling

---

## 🤝 Contributing

Contributions are welcome! Whether you are implementing a new backend adapter, optimizing CUDA tensor kernels, or enhancing the studio UI:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is dual-licensed under either:
- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

---

## ☕ Support

If you find DisposAI useful, consider buying me a coffee to support continued development:

<a href="https://buymeacoffee.com/adem.rguez" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50" style="border-radius: 8px;"></a>

