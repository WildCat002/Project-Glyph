import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ============================================================
# Glyph v2 Loss Anatomy Benchmark
# ============================================================
# Evaluates the SAME held-out validation split for:
#   - Glyph v2 20k
#   - Glyph v2 50k best
#   - Glyph v2 100k best
#
# The goal is to answer:
#   "What exactly is improving when validation loss falls?"
#
# Metrics include:
#   - overall cross-entropy / perplexity
#   - overall top-1 / top-5 accuracy
#   - loss and accuracy by target character type
#   - space/word-boundary precision, recall, F1
#   - mean predicted probability assigned to the correct class
#   - mean predicted probability of a space on non-space targets
#
# This is an evaluation-only script. It does NOT train or modify checkpoints.
# ============================================================

SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data/data.txt"
MAX_DATA = 5_000_000
VAL_CHARS = 250_000
BLOCK = 128
N_LAYER = 4
N_HEAD = 4
N_EMB = 128
DROPOUT = 0.1
EVAL_BATCH = 64
TOP_K = 5

CHECKPOINTS = [
    ("Glyph v2 20k", ROOT_DIR / "models/v2/glyph_v2.pt"),
    ("Glyph v2 50k", ROOT_DIR / "models/v2/glyph_v2_50k_best.pt"),
    ("Glyph v2 100k", ROOT_DIR / "models/v2/glyph_v2_100k_cuda_best.pt"),
]

OUTPUT_TXT = ROOT_DIR / "results/v2/glyph_v2_loss_anatomy_results.txt"
OUTPUT_CSV = ROOT_DIR / "results/v2/glyph_v2_loss_anatomy_results.csv"

# Prefer GPU when available. The benchmark also works on CPU.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    torch.set_num_threads(min(4, os.cpu_count() or 1))

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Model — exact Glyph v2 architecture
# ============================================================

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
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
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

    def forward(self, idx):
        B, T = idx.shape
        if T > BLOCK:
            raise ValueError(f"Sequence length {T} exceeds BLOCK={BLOCK}")

        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================
# Data / vocab
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing {DATA_FILE!r} in the current directory.")

with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read().lower()

text = raw_text[:MAX_DATA]
if len(text) < MAX_DATA:
    raise ValueError(
        f"data.txt has only {len(text):,} chars; expected at least {MAX_DATA:,}."
    )

TRAIN_END = MAX_DATA - VAL_CHARS
val_text = text[TRAIN_END:]

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
V = len(chars)

if V != 98:
    raise ValueError(f"Expected Glyph v2 vocab=98, reconstructed vocab={V}.")

val_data = np.fromiter((stoi[c] for c in val_text), dtype=np.int64, count=len(val_text))

# Character categories used for the anatomy analysis.
# 0 = letter, 1 = whitespace, 2 = punctuation, 3 = digit, 4 = other
cat_names = ["letter", "whitespace", "punctuation", "digit", "other"]
cat_ids_cpu = np.empty(V, dtype=np.int64)

for token_id, ch in itos.items():
    if ch.isalpha():
        cat_ids_cpu[token_id] = 0
    elif ch.isspace():
        cat_ids_cpu[token_id] = 1
    elif ch.isdigit():
        cat_ids_cpu[token_id] = 3
    elif ch.isprintable():
        cat_ids_cpu[token_id] = 2
    else:
        cat_ids_cpu[token_id] = 4

cat_ids = torch.tensor(cat_ids_cpu, dtype=torch.long, device=DEVICE)
space_id = stoi.get(" ")
if space_id is None:
    raise ValueError("Space character is missing from the reconstructed vocabulary.")

# ============================================================
# Deterministic validation iterator
# ============================================================

def validation_batches():
    """Yield contiguous, non-overlapping validation windows covering all targets."""
    n_targets = len(val_data) - 1
    starts = range(0, n_targets, BLOCK)
    batch_x = []
    batch_y = []

    for start in starts:
        end = min(start + BLOCK, n_targets)
        x = val_data[start:end]
        y = val_data[start + 1:end + 1]
        batch_x.append(x)
        batch_y.append(y)

        if len(batch_x) == EVAL_BATCH or end == n_targets:
            # All full windows in a batch have the same length. If the final
            # window is shorter, process that final batch separately.
            lengths = {len(a) for a in batch_x}
            if len(lengths) == 1:
                x_np = np.stack(batch_x)
                y_np = np.stack(batch_y)
                yield torch.from_numpy(x_np), torch.from_numpy(y_np)
                batch_x.clear()
                batch_y.clear()
            else:
                # Flush all full windows first.
                if len(batch_x) > 1:
                    x_np = np.stack(batch_x[:-1])
                    y_np = np.stack(batch_y[:-1])
                    yield torch.from_numpy(x_np), torch.from_numpy(y_np)
                # Final short window.
                yield (
                    torch.from_numpy(batch_x[-1][None, :]),
                    torch.from_numpy(batch_y[-1][None, :]),
                )
                batch_x.clear()
                batch_y.clear()


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model_state = checkpoint.get("model")
    if model_state is None:
        model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise KeyError(f"No model state found in {path}")

    model = Glyph(V).to(DEVICE)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    step = int(checkpoint.get("step", -1))
    stored_best = checkpoint.get("best_val_loss")
    if stored_best is not None:
        stored_best = float(stored_best)

    return model, step, stored_best


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(model):
    totals = {
        "tokens": 0,
        "loss_sum": 0.0,
        "top1_correct": 0,
        "top5_correct": 0,
        "correct_prob_sum": 0.0,
        "space_actual": 0,
        "space_predicted": 0,
        "space_tp": 0,
        "space_prob_actual_sum": 0.0,
        "space_prob_non_actual_sum": 0.0,
        "space_non_actual": 0,
    }

    by_cat = {
        name: {
            "count": 0,
            "loss_sum": 0.0,
            "top1_correct": 0,
            "top5_correct": 0,
            "correct_prob_sum": 0.0,
        }
        for name in cat_names
    }

    start_time = time.perf_counter()

    for x_cpu, y_cpu in validation_batches():
        x = x_cpu.to(DEVICE, non_blocking=True)
        y = y_cpu.to(DEVICE, non_blocking=True)

        logits = model(x)
        flat_logits = logits.float().reshape(-1, V)
        flat_y = y.reshape(-1)

        losses = F.cross_entropy(
            flat_logits,
            flat_y,
            reduction="none",
        )

        probs = torch.softmax(flat_logits, dim=-1)
        pred1 = flat_logits.argmax(dim=-1)
        top5 = flat_logits.topk(min(TOP_K, V), dim=-1).indices

        top1 = pred1.eq(flat_y)
        top5_hit = top5.eq(flat_y.unsqueeze(-1)).any(dim=-1)
        correct_prob = probs.gather(1, flat_y.unsqueeze(1)).squeeze(1)

        cats = cat_ids[flat_y]

        totals["tokens"] += flat_y.numel()
        totals["loss_sum"] += float(losses.sum())
        totals["top1_correct"] += int(top1.sum())
        totals["top5_correct"] += int(top5_hit.sum())
        totals["correct_prob_sum"] += float(correct_prob.sum())

        actual_space = flat_y.eq(space_id)
        predicted_space = pred1.eq(space_id)
        space_tp = actual_space & predicted_space

        p_space = probs[:, space_id]
        totals["space_actual"] += int(actual_space.sum())
        totals["space_predicted"] += int(predicted_space.sum())
        totals["space_tp"] += int(space_tp.sum())
        totals["space_prob_actual_sum"] += float(p_space[actual_space].sum()) if actual_space.any() else 0.0
        non_space = ~actual_space
        totals["space_prob_non_actual_sum"] += float(p_space[non_space].sum()) if non_space.any() else 0.0
        totals["space_non_actual"] += int(non_space.sum())

        for cat_id, name in enumerate(cat_names):
            mask = cats.eq(cat_id)
            count = int(mask.sum())
            if count == 0:
                continue
            bucket = by_cat[name]
            bucket["count"] += count
            bucket["loss_sum"] += float(losses[mask].sum())
            bucket["top1_correct"] += int(top1[mask].sum())
            bucket["top5_correct"] += int(top5_hit[mask].sum())
            bucket["correct_prob_sum"] += float(correct_prob[mask].sum())

    elapsed = time.perf_counter() - start_time

    out = {
        "tokens": totals["tokens"],
        "loss": totals["loss_sum"] / totals["tokens"],
        "perplexity": math.exp(min(20.0, totals["loss_sum"] / totals["tokens"])),
        "top1": totals["top1_correct"] / totals["tokens"],
        "top5": totals["top5_correct"] / totals["tokens"],
        "mean_correct_prob": totals["correct_prob_sum"] / totals["tokens"],
        "space_actual": totals["space_actual"],
        "space_predicted": totals["space_predicted"],
        "space_precision": (
            totals["space_tp"] / totals["space_predicted"]
            if totals["space_predicted"] else 0.0
        ),
        "space_recall": (
            totals["space_tp"] / totals["space_actual"]
            if totals["space_actual"] else 0.0
        ),
        "space_f1": 0.0,
        "space_mean_prob_when_space": (
            totals["space_prob_actual_sum"] / totals["space_actual"]
            if totals["space_actual"] else 0.0
        ),
        "space_mean_prob_when_not_space": (
            totals["space_prob_non_actual_sum"] / totals["space_non_actual"]
            if totals["space_non_actual"] else 0.0
        ),
        "elapsed_sec": elapsed,
        "tokens_per_sec": totals["tokens"] / elapsed if elapsed > 0 else 0.0,
        "by_category": {},
    }

    p = out["space_precision"]
    r = out["space_recall"]
    out["space_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    for name, bucket in by_cat.items():
        count = bucket["count"]
        out["by_category"][name] = {
            "count": count,
            "loss": bucket["loss_sum"] / count if count else 0.0,
            "top1": bucket["top1_correct"] / count if count else 0.0,
            "top5": bucket["top5_correct"] / count if count else 0.0,
            "mean_correct_prob": bucket["correct_prob_sum"] / count if count else 0.0,
        }

    return out


# ============================================================
# Reporting
# ============================================================

results = []

print("=" * 72)
print("Glyph v2 Loss Anatomy Benchmark")
print("=" * 72)
print(f"Device              : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU                 : {torch.cuda.get_device_name(0)}")
print(f"PyTorch             : {torch.__version__}")
print(f"Validation chars    : {len(val_text):,}")
print(f"Validation targets  : {len(val_data) - 1:,}")
print(f"Vocab               : {V}")
print(f"Context             : {BLOCK}")
print(f"Evaluation batch    : {EVAL_BATCH}")
print()

for name, path in CHECKPOINTS:
    print("-" * 72)
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {path}")

    model, step, stored_best = load_checkpoint(path)

    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError(f"Non-finite parameter detected in {path}")

    metrics = evaluate(model)
    metrics["model"] = name
    metrics["checkpoint"] = path
    metrics["step"] = step
    metrics["stored_best_val_loss"] = stored_best
    results.append(metrics)

    print(f"Step                  : {step:,}")
    print(f"Stored best val loss  : {stored_best}")
    print(f"Full validation loss  : {metrics['loss']:.6f}")
    print(f"Perplexity             : {metrics['perplexity']:.4f}")
    print(f"Top-1 accuracy         : {metrics['top1']:.4%}")
    print(f"Top-5 accuracy         : {metrics['top5']:.4%}")
    print(f"Mean correct prob      : {metrics['mean_correct_prob']:.4f}")
    print()
    print("Character-type anatomy")
    print(f"{'Category':<14} {'Count':>9} {'Loss':>10} {'Top1':>10} {'Top5':>10} {'P(correct)':>12}")
    for cat in cat_names:
        m = metrics["by_category"][cat]
        print(
            f"{cat:<14} {m['count']:>9,} {m['loss']:>10.6f} "
            f"{m['top1']:>10.4%} {m['top5']:>10.4%} {m['mean_correct_prob']:>12.4f}"
        )
    print()
    print("Word-boundary / space anatomy")
    print(f"Actual spaces         : {metrics['space_actual']:,}")
    print(f"Predicted spaces      : {metrics['space_predicted']:,}")
    print(f"Space precision       : {metrics['space_precision']:.4%}")
    print(f"Space recall          : {metrics['space_recall']:.4%}")
    print(f"Space F1              : {metrics['space_f1']:.4%}")
    print(f"P(space | space)      : {metrics['space_mean_prob_when_space']:.4f}")
    print(f"P(space | non-space)  : {metrics['space_mean_prob_when_not_space']:.6f}")
    print(f"Eval time             : {metrics['elapsed_sec']:.2f}s")
    print(f"Eval throughput       : {metrics['tokens_per_sec']:,.0f} target chars/s")
    print()

# ---------------------------
# Summary table
# ---------------------------

lines = []
lines.append("GLYPH V2 LOSS ANATOMY BENCHMARK")
lines.append("=" * 72)
lines.append(f"Validation: final {VAL_CHARS:,} chars of the 5M-char corpus")
lines.append(f"Targets evaluated: {len(val_data)-1:,}")
lines.append(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
lines.append("")
lines.append("SUMMARY")
lines.append("-" * 72)
lines.append(
    f"{'Model':<16} {'Step':>8} {'ValLoss':>10} {'PPL':>9} "
    f"{'Top1':>9} {'Top5':>9} {'SpaceR':>9} {'SpaceF1':>9}"
)
for m in results:
    lines.append(
        f"{m['model']:<16} {m['step']:>8,} {m['loss']:>10.6f} "
        f"{m['perplexity']:>9.3f} {m['top1']:>9.3%} {m['top5']:>9.3%} "
        f"{m['space_recall']:>9.3%} {m['space_f1']:>9.3%}"
    )

lines.append("")
lines.append("CHARACTER-TYPE LOSS")
lines.append("-" * 72)
for cat in cat_names:
    row = [cat]
    for m in results:
        row.append(
            f"{m['by_category'][cat]['loss']:.6f} "
            f"(top1 {m['by_category'][cat]['top1']:.2%})"
        )
    lines.append(f"{row[0]:<14} | " + " | ".join(row[1:]))

lines.append("")
lines.append("SPACE / WORD-BOUNDARY DIAGNOSTIC")
lines.append("-" * 72)
for m in results:
    lines.append(
        f"{m['model']}: precision={m['space_precision']:.4%}, "
        f"recall={m['space_recall']:.4%}, F1={m['space_f1']:.4%}, "
        f"P(space|space)={m['space_mean_prob_when_space']:.4f}, "
        f"P(space|non-space)={m['space_mean_prob_when_not_space']:.6f}"
    )

lines.append("")
lines.append("INTERPRETATION GUIDE")
lines.append("-" * 72)
lines.append("1. Lower overall loss/PPL means better next-character prediction on held-out validation text.")
lines.append("2. Space recall/F1 directly tests whether the model predicts word boundaries when the next character is a space.")
lines.append("3. If 100k has lower overall loss but worse space recall, some of its loss improvement is not translating into better word boundaries.")
lines.append("4. Compare letter vs whitespace vs punctuation loss to see which character classes improved most with training.")
lines.append("5. This benchmark is diagnostic; it does not produce an overall model ranking.")
lines.append("6. The stored best-validation loss is from the training script's sampled validation checks; 'Full validation loss' here is recomputed deterministically across the entire held-out split.")

report = "\n".join(lines) + "\n"
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(report)

# Flat CSV: summary + per-category rows.
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "checkpoint", "step", "stored_best_val_loss",
        "full_val_loss", "perplexity", "top1", "top5",
        "space_precision", "space_recall", "space_f1",
        "p_space_given_space", "p_space_given_nonspace",
        "category", "category_count", "category_loss",
        "category_top1", "category_top5", "category_mean_correct_prob",
    ])

    for m in results:
        for cat in cat_names:
            c = m["by_category"][cat]
            writer.writerow([
                m["model"], m["checkpoint"], m["step"], m["stored_best_val_loss"],
                m["loss"], m["perplexity"], m["top1"], m["top5"],
                m["space_precision"], m["space_recall"], m["space_f1"],
                m["space_mean_prob_when_space"], m["space_mean_prob_when_not_space"],
                cat, c["count"], c["loss"], c["top1"], c["top5"], c["mean_correct_prob"],
            ])

print("=" * 72)
print("Loss anatomy benchmark finished")
print(f"Report : {OUTPUT_TXT}")
print(f"CSV    : {OUTPUT_CSV}")
print("=" * 72)
print()
print(report)
