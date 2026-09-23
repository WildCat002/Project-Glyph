from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models/v3/glyph_v3c_best.pt"
DEFAULT_OUTPUT = ROOT / "results/v3/benchmark_v3C_results.txt"

TEMPERATURE = 0.7
TOP_K = 20
NEW_CHARS = 120
SEEDS = [42, 43, 44, 45, 46]

TASKS = [
    ("arithmetic", "user: what is 7 + 8?\nassistant:", "15"),
    ("arithmetic", "user: what is 80 - 4?\nassistant:", "76"),
    ("arithmetic", "user: what is 12 times 4?\nassistant:", "48"),
    ("arithmetic", "user: what is 81 divided by 9?\nassistant:", "9"),
    ("comparison", "user: which is larger, 30 or 87?\nassistant:", "87"),
    ("comparison", "user: which is smaller, 23 or 76?\nassistant:", "23"),
    ("sequence", "user: what comes next: 2, 5, 8, 11, ?\nassistant:", "14"),
    ("sequence", "user: what comes next: 10, 16, 22, 28, ?\nassistant:", "34"),
    ("yes_no", "user: is 17 greater than 9?\nassistant:", "yes"),
    ("yes_no", "user: is 4 less than 2?\nassistant:", "no"),
    ("string", "user: reverse the word stone.\nassistant:", "enots"),
    ("string", "user: how many letters are in apple?\nassistant:", "5"),
    ("logic", "user: all zibs are blue. alice is a zib. what color is alice?\nassistant:", "blue"),
    ("logic", "user: alice is taller than bob. bob is taller than carol. who is tallest?\nassistant:", "alice"),
    ("logic", "user: bob arrives before carol. carol arrives before david. who arrives first?\nassistant:", "bob"),
    ("conversation", "user: hello\nassistant:", "hello"),
    ("conversation", "user: thank you\nassistant:", "happy"),
    ("conversation", "user: good bye\nassistant:", "goodbye"),
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


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    chars = cfg["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    model = GlyphV3LSTM(
        len(chars),
        int(cfg.get("emb", 96)),
        int(cfg.get("hidden", 240)),
        int(cfg.get("layers", 2)),
        float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ck["model"])
    model.eval()

    return model, stoi, itos, int(cfg.get("seq_len", 128)), ck


@torch.inference_mode()
def generate(model, stoi, itos, seq_len, prompt, seed):
    torch.manual_seed(seed)

    ids = [stoi[c] for c in prompt.lower() if c in stoi]
    if not ids:
        raise ValueError(f"No known prompt characters for {prompt!r}")

    device = next(model.parameters()).device
    ids = ids[-seq_len:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    logits, hidden = model(idx)
    generated = []

    for _ in range(NEW_CHARS):
        z = logits[0, -1].float() / TEMPERATURE
        k = min(TOP_K, z.numel())
        values, indices = torch.topk(z, k)
        probs = F.softmax(values, dim=-1)
        nxt = indices[torch.multinomial(probs, 1)].item()

        generated.append(itos[nxt])

        x = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)

    return "".join(generated)


def first_answer(text):
    # Stop if the model starts fabricating another user turn.
    text = re.split(r"\n\s*user\s*:", text, maxsplit=1, flags=re.I)[0]

    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def norm(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def hit(category, expected, answer):
    a = norm(answer)
    e = norm(expected)

    if category in {"arithmetic", "comparison", "sequence"}:
        m = re.search(r"(?<!\d)-?\d+(?!\d)", a)
        return bool(m and m.group() == e)

    if category == "yes_no":
        return bool(a.split()) and a.split()[0] == e

    if category == "string":
        if e.isdigit():
            m = re.search(r"\d+", a)
            return bool(m and m.group() == e)
        return a.startswith(e)

    if category == "logic":
        return e in a[:100]

    if category == "conversation":
        return e in a[:100]

    return False


def main():
    parser = argparse.ArgumentParser(description="Glyph V3-C exact task benchmark")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, stoi, itos, seq_len, ck = load_checkpoint(args.checkpoint)
    model.to(device)

    # Generation itself is kept on the selected device.
    report = [
        "Glyph V3-C Exact Task Benchmark",
        "=" * 68,
        f"Checkpoint   : {args.checkpoint.resolve()}",
        f"Step         : {ck.get('step', 'unknown')}",
        f"Best val loss: {ck.get('best_val_loss', 'unknown')}",
        f"Parameters   : {sum(p.numel() for p in model.parameters()):,}",
        f"Device       : {device}",
        f"Temperature  : {TEMPERATURE}",
        f"Top-k        : {TOP_K}",
        f"New chars    : {NEW_CHARS}",
        f"Seeds        : {SEEDS}",
        f"Tasks        : {len(TASKS)}",
        "",
    ]

    total_hits = 0
    total_cases = 0
    category_hits = {}
    category_cases = {}

    for number, (category, prompt, expected) in enumerate(TASKS, 1):
        task_hits = 0

        for seed in SEEDS:
            answer = first_answer(
                generate(model, stoi, itos, seq_len, prompt, seed + number * 1000)
            )
            ok = hit(category, expected, answer)

            total_hits += int(ok)
            total_cases += 1
            task_hits += int(ok)

            category_hits[category] = category_hits.get(category, 0) + int(ok)
            category_cases[category] = category_cases.get(category, 0) + 1

            report.extend([
                f"TASK {number:02d} | {category:12s} | seed={seed} | "
                f"{'PASS' if ok else 'FAIL'}",
                f"prompt:   {prompt!r}",
                f"expected: {expected!r}",
                f"answer:   {answer!r}",
                "",
            ])

        report.append(
            f"TASK {number:02d} accuracy: "
            f"{task_hits}/{len(SEEDS)} = {task_hits / len(SEEDS):.4f}"
        )
        report.append("")

    report.extend([
        "=" * 68,
        "CATEGORY RESULTS",
        "=" * 68,
    ])

    for category in category_hits:
        h = category_hits[category]
        n = category_cases[category]
        report.append(f"{category:14s}: {h}/{n} = {h / n:.4f}")

    report.extend([
        "",
        "=" * 68,
        f"Expected-answer hit rate: {total_hits / total_cases:.4f}",
        f"Total correct: {total_hits}/{total_cases}",
        "=" * 68,
        "",
        "Method notes:",
        "1. Every task is sampled with five deterministic seeds.",
        "2. Only the first non-empty assistant answer is scored.",
        "3. Fabricated later user/assistant turns are excluded.",
        "4. Arithmetic/comparison/sequence require the expected number.",
        "5. Yes/no requires the expected first token.",
        "6. String tasks require the expected exact prefix/value.",
        "7. Logic/conversation tasks use a bounded answer match.",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report), encoding="utf-8")

    print("=" * 68)
    print(f"Benchmark complete. Results saved to: {args.output.resolve()}")
    print(f"Expected-answer hit rate: {total_hits / total_cases:.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
