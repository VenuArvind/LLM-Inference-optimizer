# LLM-Inference-optimizer

Running useful LLMs under tight memory & latency constraints on Apple Silicon.

## Project Overview

This project implements an optimized on-device LLM inference engine for Apple Silicon (M2/M3), featuring:
- Manual inference loop (without `model.generate()`)
- Quantization (INT8/INT4)
- KV-Cache optimization
- Speculative decoding
- CoreML integration

## Quick Start

1. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Project Structure

```
LLM-Inference-optimizer/
├── src/              # Engine implementations
├── benchmarks/       # Benchmarking scripts
├── utils/            # Utility functions
└── results/          # Benchmark results and plots
```

## Results