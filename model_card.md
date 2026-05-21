---
language:
  - en
license: mit
tags:
  - small-language-model
  - edge-ai
  - decoder-only
  - tiny-transformer
  - pytorch
  - onnx
---

# KRULL-Nano Simple

KRULL-Nano Simple is a minimal decoder-only Small Language Model architecture designed for educational experiments with edge-oriented language modeling.

KRULL means **Knowledge Running Under Lightweight Language**.

This implementation prioritizes simplicity, portability and low resource usage. It is intended as a clean baseline for building small language models that can later be optimized for embedded and edge devices.

## Architecture

- Decoder-only transformer
- RMSNorm
- Multi-query attention
- Gated feed-forward network
- Character-level tokenizer
- CPU-first training
- ONNX export support

## Intended Use

- Learning how small decoder-only language models work
- Edge AI prototyping
- Offline language modeling experiments
- Lightweight autoregressive generation research

## Limitations

The included dataset is tiny and only intended to verify that the pipeline works. The model must be trained on a larger dataset for meaningful generation.

## License

MIT
