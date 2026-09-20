import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data/data.txt"
CHECKPOINTS = [("best", ROOT_DIR / "models/v2/glyph_v2_best.pt"), ("final", ROOT_DIR / "models/v2/glyph_v2.pt")]
MAX_DATA = 5_000_000
VAL_CHARS = 250_000
BLOCK = 128
N_LAYER = 4
N_HEAD = 4
N_EMB = 128
DROPOUT = 0.1
VAL_BATCH = 32
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
    "What",
]

torch.set_num_threads(4)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMB, 3 * N_EMB, bias=False)
        self.proj = nn.Linear(N_EMB, N_EMB, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMB, dim=2)
        head_dim = C // N_HEAD
        q = q.view(B, T, N_HEAD, head_dim).transpose(1, 2).float()
        k = k.view(B, T, N_HEAD, head_dim).transpose(1, 2).float()
        v = v.view(B, T, N_HEAD, head_dim).transpose(1, 2).float()
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMB)
        self.attn = Attn()
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
        self.head.weight = self.tok.weight

    def forward(self, idx):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def load_data():
    path = Path(DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}")
    text = path.read_text(encoding="utf-8", errors="ignore").lower()[:MAX_DATA]
    if len(text) <= VAL_CHARS + BLOCK:
        raise ValueError("Corpus is too small for the configured validation split.")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    val_text = text[len(text) - VAL_CHARS:]
    val_ids = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
    return text, val_text, val_ids, stoi, itos


def load_checkpoint(path, vocab_size):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Glyph(vocab_size)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, checkpoint.get("step", "unknown"), checkpoint.get("best_val_loss")


@torch.no_grad()
def evaluate_validation(model, val_ids):
    model.eval()
    losses = []
    bx, by = [], []
    t0 = time.time()
    limit = len(val_ids) - BLOCK - 1
    for s in range(0, limit + 1, BLOCK):
        bx.append(val_ids[s:s + BLOCK])
        by.append(val_ids[s + 1:s + BLOCK + 1])
        if len(bx) == VAL_BATCH:
            x, y = torch.stack(bx), torch.stack(by)
            logits = model(x).float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            losses.append(loss.item())
            bx.clear(); by.clear()
    if bx:
        x, y = torch.stack(bx), torch.stack(by)
        logits = model(x).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss, math.exp(min(loss, 20.0)), time.time() - t0, len(losses)


@torch.no_grad()
def generate(model, stoi, itos, prompt, max_new_tokens, temperature, top_k):
    normalized_prompt = prompt.lower()
    ids = [stoi[c] for c in normalized_prompt if c in stoi]
    if not ids:
        raise ValueError(f"Prompt contains no known characters: {prompt!r}")
    idx = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -BLOCK:])[:, -1, :].float()
        logits = logits / max(temperature, 1e-6)
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k=k)
        probs = F.softmax(values, dim=-1)
        next_pos = torch.multinomial(probs, 1)
        next_token = indices.gather(-1, next_pos)
        idx = torch.cat((idx, next_token), dim=1)
    return "".join(itos[int(i)] for i in idx[0]), normalized_prompt, time.time() - t0


def generation_stats(text):
    words = re.findall(r"[a-zA-Z]+", text)
    char_repeat = len(re.findall(r"(.)\1{3,}", text))
    repeated_words = sum(words[i] == words[i - 1] for i in range(1, len(words)))
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    repeated_trigrams = len(trigrams) - len(set(trigrams)) if trigrams else 0
    printable = sum(c.isprintable() or c in "\n\t" for c in text)
    return {
        "characters": len(text),
        "words": len(words),
        "printable_pct": 100 * printable / max(len(text), 1),
        "repeat_chars": char_repeat,
        "repeat_words": repeated_words,
        "repeat_trigrams": repeated_trigrams,
    }


def memorization_signal(model, stoi, itos, corpus):
    results = []
    for prompt in PROMPTS:
        set_seed(SEED)
        output, normalized, _ = generate(model, stoi, itos, prompt, GEN_TOKENS, 0.9, TOP_K)
        tail = output[len(normalized):]
        chunks = [tail[i:i + 20] for i in range(0, min(max(len(tail) - 19, 0), 200), 20)]
        matches = sum(chunk in corpus for chunk in chunks if len(chunk) == 20)
        results.append((prompt, matches, len(chunks)))
    return results


def main():
    set_seed(SEED)
    print("=" * 64)
    print("Glyph v2 Benchmark")
    print("=" * 64)
    corpus, val_text, val_ids, stoi, itos = load_data()
    print(f"Corpus chars      : {len(corpus):,}")
    print(f"Validation chars  : {len(val_text):,}")
    print(f"Vocabulary        : {len(stoi)}")
    print(f"Context           : {BLOCK}")
    print(f"Layers            : {N_LAYER}")
    print()
    report = [
        "Glyph v2 Benchmark",
        "=" * 64,
        f"Corpus chars     : {len(corpus):,}",
        f"Validation chars : {len(val_text):,}",
        f"Vocabulary       : {len(stoi)}",
        "",
    ]
    for label, ckpt_path in CHECKPOINTS:
        print("=" * 64)
        print(f"Checkpoint: {ckpt_path}")
        print("=" * 64)
        path = Path(ckpt_path)
        if not path.exists():
            print(f"SKIPPED: {ckpt_path} not found.\n")
            continue
        model, checkpoint, step, saved_best = load_checkpoint(path, len(stoi))
        param_count = sum(p.numel() for p in model.parameters())
        finite = all(torch.isfinite(p).all().item() for p in model.parameters())
        print(f"Checkpoint step   : {step}")
        print(f"Saved best val    : {saved_best}")
        print(f"Parameters        : {param_count:,}")
        print(f"Checkpoint finite : {'YES' if finite else 'NO'}")
        report += [
            "", f"CHECKPOINT: {ckpt_path}", "-" * 64,
            f"Checkpoint step   : {step}",
            f"Saved best val    : {saved_best}",
            f"Parameters        : {param_count:,}",
            f"Checkpoint finite : {'YES' if finite else 'NO'}",
        ]
        print("\nFull validation evaluation...")
        val_loss, ppl, eval_time, batches = evaluate_validation(model, val_ids)
        print(f"Validation loss   : {val_loss:.4f}")
        print(f"Perplexity        : {ppl:.2f}")
        print(f"Validation time   : {eval_time:.1f}s")
        print(f"Validation batches: {batches}")
        report += [
            "", "Validation", "-" * 64,
            f"Validation loss   : {val_loss:.6f}",
            f"Perplexity        : {ppl:.4f}",
            f"Validation time   : {eval_time:.1f}s",
            f"Validation batches: {batches}",
            "", "Generation", "-" * 64,
        ]
        for temperature in GEN_TEMPERATURES:
            print(f"\nTemperature: {temperature}")
            report += ["", f"Temperature: {temperature}"]
            for prompt in PROMPTS:
                set_seed(SEED)
                output, normalized, gen_time = generate(model, stoi, itos, prompt, GEN_TOKENS, temperature, TOP_K)
                stats = generation_stats(output)
                generated_chars = max(len(output) - len(normalized), 1)
                speed = generated_chars / gen_time if gen_time > 0 else 0.0
                print(f"\nPROMPT: {prompt!r}")
                print(output)
                print(
                    f"stats: words={stats['words']} repeat_chars={stats['repeat_chars']} "
                    f"repeat_words={stats['repeat_words']} repeat_trigrams={stats['repeat_trigrams']} "
                    f"speed={speed:.1f} char/s"
                )
                report += [
                    "", f"PROMPT: {prompt!r}", output,
                    "Stats: "
                    f"words={stats['words']}, printable={stats['printable_pct']:.2f}%, "
                    f"repeat_chars={stats['repeat_chars']}, repeat_words={stats['repeat_words']}, "
                    f"repeat_trigrams={stats['repeat_trigrams']}, generation_speed={speed:.2f} char/s",
                ]
        print("\nSimple memorization signal...")
        report += ["", "Simple memorization signal", "-" * 64]
        for prompt, matches, total in memorization_signal(model, stoi, itos, corpus):
            line = f"{prompt!r}: {matches}/{total} 20-char chunks found in corpus"
            print(line)
            report.append(line)
    report += ["", "=" * 64, "End of benchmark"]
    out = ROOT_DIR / "results/v2/benchmark_v2_results.txt"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n" + "=" * 64)
    print(f"Benchmark complete. Results saved to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
