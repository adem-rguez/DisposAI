# DisposAI TypeScript / JavaScript SDK

Official TypeScript client for the **DisposAI** local inference daemon.

## Installation

```bash
npm install dispos-sdk
```

## Quick Start

```typescript
import { DisposClient } from 'dispos-sdk';

const client = new DisposClient('http://localhost:8080');

async function run() {
  // 1. Check if model fits in VRAM/RAM
  const fit = await client.estimateFit({
    parameter_count_billions: 8.0,
    quantization: 'Q4_K_M',
    context_size: 4096,
  });
  console.log(`Fits in GPU: ${fit.fits_in_vram}`);

  // 2. Chat completion
  const completion = await client.chatCompletion({
    model: 'models/Qwen3.5-0.8B-Q8_0.gguf',
    messages: [{ role: 'user', content: 'Hello from TypeScript!' }],
  });
  console.log(completion.choices[0].message.content);
}

run();
```

## License

MIT OR Apache-2.0
