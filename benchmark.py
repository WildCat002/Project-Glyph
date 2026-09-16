"""
Glyph v1 Benchmark
------------------
Benchmarks the trained glyph.pt checkpoint without modifying it.

Run:
    python benchmark.py

Output:
    benchmark_results.txt
"""

import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ============================================================
# Configuration — must match Glyph v1
# ============================================================

BLOCK = 64
N_LAYER = 3
N_HEAD = 4
N_EMB = 128
DROPOUT = 0.1

CKPT = "glyph.pt"
DATA = "data.txt"

EVAL_CHARS = 100_000
EVAL_BATCH = 16
TRAIN_CHARS = 5_000_000

GEN_TOKENS = 400
GEN_TEMPERATURES = [0.7, 0.9, 1.0]
TOP_K = 20

SEED = 42

PROMPTS = [
    "Alice was",
    "Alice was sitting",
    "The Queen said",
    "Alice looked",
    "The rabbit",
    "Once upon a time",
    "“What",
]


# ============================================================
# Model
# ============================================================


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMB, 3 * N_EMB, bias=False)
        self.proj = nn.Linear(N_EMB, N_EMB, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMB, dim=2)

        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2).float()
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2).float()
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2).float()

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMB)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMB)

        self.mlp = nn.Sequential(
            nn.Linear(N_EMB, 4 * N_EMB),
            nn.GELU(),
            nn.Linear(4 * N_EMB, N_EMB),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Glyph(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.tok = nn.Embedding(vocab_size, N_EMB)
        self.pos = nn.Embedding(BLOCK, N_EMB)

        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])

        self.ln_f = nn.LayerNorm(N_EMB)
        self.head = nn.Linear(N_EMB, vocab_size, bias=False)

        # Glyph uses tied token/output weights.
        self.head.weight = self.tok.weight

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)

        x = self.blocks(x)
        x = self.ln_f(x)

        return self.head(x)


# ============================================================
# Helpers
# ============================================================


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_project():
    data_path = Path(DATA)
    ckpt_path = Path(CKPT)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {DATA}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {CKPT}")

    # Match Glyph training: lowercase and use sorted character vocab.
    text = data_path.read_text(encoding="utf-8").lower()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    model = Glyph(len(chars))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    step = checkpoint.get("step", "unknown")

    return text, stoi, itos, model, step


def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]


def generate(model, stoi, itos, prompt, max_new_tokens, temperature, top_k):
    ids = encode(prompt, stoi)

    if not ids:
        raise ValueError(f"Prompt contains no known characters: {prompt!r}")

    idx = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK:]
            logits = model(idx_cond)[:, -1, :]
            logits = logits.float() / max(temperature, 1e-6)

            k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, k=k)

            probs = F.softmax(values, dim=-1)
            next_pos = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, next_pos)

            idx = torch.cat((idx, next_token), dim=1)

    return "".join(itos[int(i)] for i in idx[0])


def repetition_stats(text):
    # Repeated characters, e.g. "aaaaa"
    char_repeat = len(re.findall(r"(.)\1{3,}", text))

    # Repeated short words/phrases.
    words = re.findall(r"[a-zA-Z]+", text.lower())

    repeated_words = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeated_words += 1

    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]

    repeated_trigrams = len(trigrams) - len(set(trigrams)) if trigrams else 0

    return char_repeat, repeated_words, repeated_trigrams


def generation_stats(text):
    chars = len(text)
    words = re.findall(r"[a-zA-Z]+", text)

    printable = sum(c.isprintable() or c in "\n\t" for c in text)
    printable_pct = 100 * printable / max(chars, 1)

    spaces = text.count(" ")
    punctuation = sum(c in ".,!?;:'\"()-" for c in text)

    char_repeat, repeated_words, repeated_trigrams = repetition_stats(text)

    return {
        "characters": chars,
        "words": len(words),
        "printable_pct": printable_pct,
        "space_pct": 100 * spaces / max(chars, 1),
        "punctuation_pct": 100 * punctuation / max(chars, 1),
        "char_repeat_sequences": char_repeat,
        "repeated_adjacent_words": repeated_words,
        "repeated_trigrams": repeated_trigrams,
    }


def evaluate_loss(model, ids, start, length, batch_size):
    # Evaluate a contiguous held-out section of the corpus.
    available = length - BLOCK - 1

    if available <= 0:
        raise ValueError("Evaluation data is too small.")

    losses = []

    with torch.no_grad():
        for pos in range(start, start + available, batch_size):
            starts = list(
                range(
                    pos,
                    min(pos + batch_size, start + available),
                )
            )

            x = torch.stack([ids[s : s + BLOCK] for s in starts])

            y = torch.stack([ids[s + 1 : s + BLOCK + 1] for s in starts])

            logits = model(x).float()

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            losses.append(float(loss))

    loss = sum(losses) / len(losses)
    perplexity = math.exp(min(loss, 20))

    return loss, perplexity


# ============================================================
# Main benchmark
# ============================================================


def main():
    set_seed(SEED)

    print("=" * 60)
    print("Glyph v1 Benchmark")
    print("=" * 60)

    text, stoi, itos, model, step = load_project()

    param_count = sum(p.numel() for p in model.parameters())

    print(f"Checkpoint step : {step}")
    print(f"Parameters      : {param_count:,}")
    print(f"Vocabulary      : {len(stoi)}")
    print(f"Corpus chars    : {len(text):,}")
    print()

    results = []
    report = []

    report.append("Glyph v1 Benchmark")
    report.append("=" * 60)
    report.append(f"Checkpoint step : {step}")
    report.append(f"Parameters      : {param_count:,}")
    report.append(f"Vocabulary      : {len(stoi)}")
    report.append(f"Corpus chars    : {len(text):,}")
    report.append(f"Training chars  : {TRAIN_CHARS:,}")
    report.append("Attention qkv/proj bias: disabled (matches glyph.pt)")
    report.append("")

    # --------------------------------------------------------
    # Checkpoint sanity
    # --------------------------------------------------------

    finite = all(torch.isfinite(p).all().item() for p in model.parameters())

    print(f"Checkpoint finite: {'YES' if finite else 'NO'}")
    report.append(f"Checkpoint finite: {'YES' if finite else 'NO'}")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    # Glyph v1 trained on the first TRAIN_CHARS characters.
    # Only call the tail "held-out" if data.txt actually contains
    # more than TRAIN_CHARS characters. If the corpus is exactly
    # 5,000,000 chars, the tail was part of training and is reported
    # as "training-overlap evaluation" instead.
    eval_start = max(0, len(text) - EVAL_CHARS)

    if len(text) > TRAIN_CHARS:
        eval_start = max(TRAIN_CHARS, len(text) - EVAL_CHARS)
        eval_label = "Held-out evaluation"
    else:
        eval_start = max(0, len(text) - EVAL_CHARS)
        eval_label = "Training-overlap evaluation"

    eval_text = text[eval_start:]
    eval_ids_list = encode(eval_text, stoi)

    ids = torch.tensor(eval_ids_list, dtype=torch.long)

    print()
    print(f"Evaluating {eval_label.lower()}...")

    t0 = time.time()
    eval_loss, perplexity = evaluate_loss(
        model,
        ids,
        start=0,
        length=len(ids),
        batch_size=EVAL_BATCH,
    )
    elapsed = time.time() - t0

    print(
        f"{'Held-out' if eval_label == 'Held-out evaluation' else 'Evaluation'} loss : {eval_loss:.4f}"
    )
    print(f"Perplexity    : {perplexity:.2f}")
    print(f"Eval time     : {elapsed:.1f}s")

    report.append("")
    report.append(eval_label)
    report.append("-" * 60)
    report.append(f"Evaluation chars: {len(eval_text):,}")
    report.append(f"Evaluation range: {eval_start:,} - {eval_start + len(eval_text):,}")
    report.append(f"Loss            : {eval_loss:.4f}")
    report.append(f"Perplexity      : {perplexity:.2f}")
    report.append(f"Evaluation time : {elapsed:.1f}s")

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    report.append("")
    report.append("Generation tests")
    report.append("-" * 60)

    for temperature in GEN_TEMPERATURES:
        print()
        print("=" * 60)
        print(f"Temperature: {temperature}")
        print("=" * 60)

        report.append("")
        report.append(f"Temperature: {temperature}")

        for prompt in PROMPTS:
            set_seed(SEED)

            output = generate(
                model,
                stoi,
                itos,
                prompt,
                GEN_TOKENS,
                temperature,
                TOP_K,
            )

            stats = generation_stats(output)

            print()
            print(f"PROMPT: {prompt!r}")
            print(output)
            print()
            print(
                f"chars={stats['characters']} "
                f"words={stats['words']} "
                f"repeat_chars={stats['char_repeat_sequences']} "
                f"repeat_words={stats['repeated_adjacent_words']} "
                f"repeat_trigrams={stats['repeated_trigrams']}"
            )

            report.append("")
            report.append(f"PROMPT: {prompt!r}")
            report.append(output)
            report.append("")
            report.append(
                "Stats: "
                f"chars={stats['characters']}, "
                f"words={stats['words']}, "
                f"printable={stats['printable_pct']:.2f}%, "
                f"repeat_chars={stats['char_repeat_sequences']}, "
                f"repeat_words={stats['repeated_adjacent_words']}, "
                f"repeat_trigrams={stats['repeated_trigrams']}"
            )

    # --------------------------------------------------------
    # Basic corpus overlap / memorization signal
    # --------------------------------------------------------

    # Look for generated 20-character sequences in the corpus.
    # This is only a simple signal, not a definitive memorization test.
    print()
    print("=" * 60)
    print("Simple memorization check")
    print("=" * 60)

    report.append("")
    report.append("Simple memorization check")
    report.append("-" * 60)

    for prompt in PROMPTS:
        set_seed(SEED)

        output = generate(
            model,
            stoi,
            itos,
            prompt,
            GEN_TOKENS,
            0.9,
            TOP_K,
        )

        generated_tail = output[len(prompt) :]

        chunks = []
        if len(generated_tail) >= 20:
            for i in range(0, min(len(generated_tail) - 19, 200), 20):
                chunks.append(generated_tail[i : i + 20])

        matches = sum(chunk in text for chunk in chunks if len(chunk) == 20)

        total = len(chunks)

        line = f"{prompt!r}: {matches}/{total} " "20-char chunks found in corpus"

        print(line)
        report.append(line)

    report.append("")
    report.append("=" * 60)
    report.append("End of benchmark")

    output_path = Path("benchmark_results.txt")
    output_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("Benchmark complete.")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
