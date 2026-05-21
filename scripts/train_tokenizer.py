import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from krull import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/tiny_corpus.txt')
    parser.add_argument('--out', default='artifacts/tokenizer.json')
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding='utf-8')
    tok = CharTokenizer()
    tok.train(text)
    tok.save(args.out)
    print(f'Tokenizer saved to {args.out}')
    print(f'Vocab size: {tok.vocab_size}')


if __name__ == '__main__':
    main()
