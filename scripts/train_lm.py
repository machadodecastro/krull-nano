import argparse
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(1)
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from krull import CharTokenizer, KRULLConfig, KRULLNano


class TextDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, i):
        x = self.ids[i:i+self.block_size]
        y = self.ids[i+1:i+self.block_size+1]
        return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/krull_nano.json')
    p.add_argument('--tokenizer', default='artifacts/tokenizer.json')
    p.add_argument('--data', default='data/tiny_corpus.txt')
    p.add_argument('--out', default='artifacts/krull_nano.pt')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    tok = CharTokenizer.load(args.tokenizer)
    cfg_data = json.loads(Path(args.config).read_text(encoding='utf-8'))
    cfg = KRULLConfig(vocab_size=tok.vocab_size, **cfg_data)
    model = KRULLNano(cfg).to(args.device)

    text = Path(args.data).read_text(encoding='utf-8')
    ids = tok.encode(text)
    ds = TextDataset(ids, cfg.block_size)
    if len(ds) == 0:
        raise RuntimeError('Dataset is too small for the configured block_size.')
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, y in dl:
            x, y = x.to(args.device), y.to(args.device)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f'epoch {epoch+1}/{args.epochs} loss={total/len(dl):.4f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': cfg.__dict__, 'model': model.state_dict()}, args.out)
    print(f'Model saved to {args.out}')


if __name__ == '__main__':
    main()
