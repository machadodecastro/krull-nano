import argparse
import sys
from pathlib import Path

import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from krull import KRULLConfig, KRULLNano


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='artifacts/krull_nano.pt')
    p.add_argument('--out', default='artifacts/krull_nano.onnx')
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location='cpu')
    cfg = KRULLConfig(**ckpt['config'])
    model = KRULLNano(cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy = torch.zeros(1, min(16, cfg.block_size), dtype=torch.long)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=['input_ids'],
        output_names=['logits', 'loss'],
        opset_version=17,
        dynamic_axes={'input_ids': {1: 'seq'}, 'logits': {1: 'seq'}},
    )
    print(f'ONNX exported to {args.out}')


if __name__ == '__main__':
    main()
