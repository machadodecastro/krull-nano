import json
from pathlib import Path


class CharTokenizer:
    """Simple robust tokenizer for KRULL-Nano.

    It is intentionally character-level to avoid BPE/training complexity.
    This makes the project easy to run on Windows, Linux and macOS.
    """
    def __init__(self, stoi=None, itos=None):
        self.stoi = stoi or {}
        self.itos = itos or {}
        self.unk_token = "<unk>"

    def train(self, text: str):
        chars = sorted(set(text))
        vocab = [self.unk_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str):
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), self.unk_token) for i in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stoi": self.stoi}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = data["stoi"]
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)
