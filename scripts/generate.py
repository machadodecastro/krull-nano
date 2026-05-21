import argparse
import sys
from pathlib import Path

import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from krull import CharTokenizer, KRULLConfig, KRULLNano


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='artifacts/krull_nano.pt')
    p.add_argument('--tokenizer', default='artifacts/tokenizer.json')
    p.add_argument('--prompt', default='KRULL is')
    p.add_argument('--max-new-tokens', type=int, default=120)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    tok = CharTokenizer.load(args.tokenizer)
    ckpt = torch.load(args.model, map_location=args.device)
    cfg = KRULLConfig(**ckpt['config'])
    model = KRULLNano(cfg).to(args.device)
    model.load_state_dict(ckpt['model'])

    x = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)
    y = model.generate(x, max_new_tokens=args.max_new_tokens)
    print(tok.decode(y[0].tolist()))


if __name__ == '__main__':
    main()
