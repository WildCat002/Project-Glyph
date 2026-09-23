"""Small reproducible benchmark for Glyph v3-A checkpoints."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT_DIR / "models/v3/glyph_v3_lstm_best.pt"
PROMPTS = [
    "alice was",
    "the king",
    "the queen",
    "once upon a time",
    "she looked at the",
    "the man",
    "user: hello\nassistant:",
    "user: what is 2 + 2?\nassistant:",
    "user: explain gravity in one sentence.\nassistant:",
]


class GlyphV3LSTM(nn.Module):
    def __init__(self, vocab_size, emb, hidden, layers, dropout):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, emb)
        self.rnn = nn.LSTM(
            emb,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, hidden=None):
        x, hidden = self.rnn(self.tok(idx), hidden)
        return self.head(x), hidden


def sample_next(logits, temperature, top_k):
    logits = logits / temperature
    if top_k:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        filtered = torch.full_like(logits, float("-inf"))
        filtered[indices] = values
        logits = filtered
    return torch.multinomial(F.softmax(logits, dim=-1), 1).item()


@torch.inference_mode()
def generate(
    model,
    text,
    stoi,
    itos,
    device,
    seq_len,
    chars,
    temperature=0.7,
    top_k=20,
    max_new=300,
):
    prompt = text.lower()
    ids = [stoi[ch] for ch in prompt if ch in stoi][-seq_len:]
    if not ids:
        ids = [stoi[" "]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits, hidden = model(idx)
    out = []
    for _ in range(max_new):
        nxt = sample_next(logits[0, -1], temperature, top_k)
        out.append(itos[nxt])
        current = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, hidden = model(current, hidden)
    return "".join(out)


def structural_metrics(text, reference_words):
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
    valid = [w for w in words if w in reference_words]
    malformed = len(words) - len(valid)
    repeated_words = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return {
        "words": len(words),
        "valid_ratio": len(valid) / max(len(words), 1),
        "malformed_ratio": malformed / max(len(words), 1),
        "repeat_adjacent": repeated_words / max(len(words) - 1, 1),
        "punctuation_ratio": sum(ch in ".,!?;:" for ch in text) / max(len(text), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    chars = cfg["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    model = GlyphV3LSTM(
        len(chars),
        int(cfg["emb"]),
        int(cfg["hidden"]),
        int(cfg["layers"]),
        float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # A lightweight in-project reference vocabulary from the same corpus.
    data = (
        (ROOT_DIR / "data/data.txt")
        .read_text(encoding="utf-8", errors="ignore")
        .lower()
    )
    reference_words = set(re.findall(r"[a-z]+", data))

    report = [
        "=" * 68,
        "Glyph v3-A Benchmark",
        "=" * 68,
        f"Checkpoint   : {args.checkpoint}",
        f"Step         : {ckpt.get('step', '?')}",
        f"Best val loss: {ckpt.get('best_val_loss', float('nan')):.6f}",
        f"Parameters   : {sum(p.numel() for p in model.parameters()):,}",
        f"Device       : {device}",
        "=" * 68,
    ]

    valid_ratios = []

    for prompt in PROMPTS:
        completion = generate(
            model,
            prompt,
            stoi,
            itos,
            device,
            int(cfg["seq_len"]),
            chars,
        )

        metrics = structural_metrics(completion, reference_words)
        valid_ratios.append(metrics["valid_ratio"])

        report.extend(
            [
                "",
                f"--- {prompt!r} ---",
                completion,
                (
                    f"metrics: valid={metrics['valid_ratio']:.3f} "
                    f"malformed={metrics['malformed_ratio']:.3f} "
                    f"adj_repeat={metrics['repeat_adjacent']:.3f} "
                    f"punct={metrics['punctuation_ratio']:.3f}"
                ),
            ]
        )

    report.extend(
        [
            "",
            f"Aggregate valid-word ratio: {statistics.mean(valid_ratios):.4f}",
            "",
            "=" * 64,
            "End of benchmark",
        ]
    )

    out = ROOT_DIR / "results/v3/benchmark_v3A_results.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"Benchmark complete. Results saved to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
