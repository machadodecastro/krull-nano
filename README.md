# KRULL-Nano Simple

**KRULL** means **Knowledge Running Under Lightweight Language**.

This is a simple, clean, CPU-first KRULL-Nano project. It is intentionally small and easy to run.

It avoids fragile features such as multiprocessing, complex BPE training, CUDA assumptions, and package installation problems.

## What is included

- Decoder-only GPT-style model
- RMSNorm
- Multi-query attention
- Gated feed-forward block
- Simple character tokenizer
- CPU training script
- Text generation script
- ONNX export script
- Windows-friendly imports

## Project structure

```text
krull_nano_simple/
├── krull/
│   ├── __init__.py
│   ├── model.py
│   └── tokenizer.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── train_lm.py
│   ├── generate.py
│   └── export_onnx.py
├── configs/
│   └── krull_nano.json
├── data/
│   └── tiny_corpus.txt
├── artifacts/
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup on Windows

Open PowerShell or CMD:

```bash
cd C:\workspace\krull_nano_simple
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Setup on Linux/macOS

```bash
cd krull_nano_simple
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Train tokenizer

```bash
python scripts/train_tokenizer.py --input data/tiny_corpus.txt --out artifacts/tokenizer.json
```

## 2. Train model

```bash
python scripts/train_lm.py --config configs/krull_nano.json --tokenizer artifacts/tokenizer.json --data data/tiny_corpus.txt --out artifacts/krull_nano.pt --epochs 10 --device cpu
```

## 3. Generate text

```bash
python scripts/generate.py --model artifacts/krull_nano.pt --tokenizer artifacts/tokenizer.json --prompt "KRULL is" --device cpu
```

## 4. Export ONNX

```bash
python scripts/export_onnx.py --model artifacts/krull_nano.pt --out artifacts/krull_nano.onnx
```

## Notes

This repo is for learning and experimentation. The default dataset is tiny, so the generated text will not be intelligent. Replace `data/tiny_corpus.txt` with a larger corpus to train a better model.

## Why character tokenizer?

The first priority here is a working, simple KRULL-Nano baseline. Character tokenization avoids common tokenizer training issues on Windows and keeps the repo easy to understand.

A BPE or map-reduce tokenizer can be added later after the base model is stable.

## License

MIT
