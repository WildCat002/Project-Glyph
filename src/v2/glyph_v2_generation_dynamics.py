import math
import random
import time
from pathlib import Path
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Glyph v2 Generation Dynamics Benchmark — FIXED
#
# Purpose:
#   1) Compare teacher-forced validation behavior.
#   2) Measure free-running generation drift.
#   3) Compare greedy vs top-k sampling.
#
# IMPORTANT METHODOLOGY:
#   - Fixed prompts are used ONLY for generation-quality tests.
#   - Drift/ground-truth tests use automatically selected anchors
#     from the held-out validation split so the reference continuation
#     is real held-out text.
#
# This avoids the earlier bug where "alice was", etc. were searched
# inside the final 250k validation slice and often were not present.
# ============================================================


SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[2]
MAX_DATA = 5_000_000
VAL_CHARS = 250_000
TRAIN_END = MAX_DATA - VAL_CHARS

BLOCK = 128
N_LAYER = 4
N_HEAD = 4
N_EMB = 128
TOP_K = 20

GENERATE_CHARS = 500
HORIZONS = [0, 10, 25, 50, 100, 250, 500]

FIXED_PROMPTS = [
    "alice was",
    "the rabbit",
    "once upon a time",
    "the king",
    "she looked",
    "the man",
    "what",
]

MODELS = {
    "Glyph v2 20k": ROOT_DIR / "models/v2/glyph_v2.pt",
    "Glyph v2 50k": ROOT_DIR / "models/v2/glyph_v2_50k_best.pt",
    "Glyph v2 100k": ROOT_DIR / "models/v2/glyph_v2_100k_cuda_best.pt",
}

TEMPS = [0.7, 0.9, 1.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_TXT = ROOT_DIR / "results/v2/glyph_v2_generation_dynamics_fixed_results.txt"
OUT_CSV = ROOT_DIR / "results/v2/glyph_v2_generation_dynamics_fixed_results.csv"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


print("=" * 72)
print("GLYPH V2 GENERATION DYNAMICS BENCHMARK — FIXED")
print("=" * 72)
print(f"Device       : {DEVICE}")
print(f"PyTorch      : {torch.__version__}")
if DEVICE.type == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
print("=" * 72)


# ============================================================
# Data / vocabulary
# ============================================================

with open(ROOT_DIR / "data/data.txt", "r", encoding="utf-8") as f:
    text = f.read()[:MAX_DATA].lower()

chars = sorted(set(text))
if len(chars) != 98:
    raise RuntimeError(f"Expected vocab=98, found {len(chars)}")

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

train_text = text[:TRAIN_END]
val_text = text[TRAIN_END:MAX_DATA]

val_ids_cpu = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)

print(f"Total chars  : {len(text):,}")
print(f"Train chars  : {len(train_text):,}")
print(f"Val chars    : {len(val_text):,}")
print(f"Vocab        : {len(chars)}")


# ============================================================
# Model
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMB, 3 * N_EMB, bias=False)
        self.proj = nn.Linear(N_EMB, N_EMB, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(N_EMB, dim=-1)
        hd = N_EMB // N_HEAD
        q = q.view(b, t, N_HEAD, hd).transpose(1, 2)
        k = k.view(b, t, N_HEAD, hd).transpose(1, 2)
        v = v.view(b, t, N_HEAD, hd).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
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
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Glyph(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(98, N_EMB)
        self.pos = nn.Embedding(BLOCK, N_EMB)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMB)
        self.head = nn.Linear(N_EMB, 98, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        b, t = idx.shape
        if t > BLOCK:
            idx = idx[:, -BLOCK:]
            t = BLOCK
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.blocks(x)
        return self.head(self.ln_f(x))


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = Glyph().to(DEVICE)
    state = ckpt.get("model", ckpt.get("model_state_dict"))
    if state is None:
        raise RuntimeError(f"{path}: model state not found")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt.get("step", "?"), ckpt.get("best_val_loss")


def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=DEVICE)


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


# ============================================================
# Full held-out validation loss — vectorized
# ============================================================

@torch.inference_mode()
def full_val_loss(model):
    # Evaluate all validation targets in non-overlapping windows.
    total_nll = 0.0
    total_n = 0

    # First BLOCK-1 tokens cannot all be predicted with a preceding
    # context inside the held-out segment. Start with a zero-length
    # boundary and then evaluate the remaining sequence.
    ids = val_ids_cpu.to(DEVICE)

    # Use windows with overlap of one token so every target from
    # position 1 onward is scored exactly once with available context.
    for start in range(0, len(ids) - 1, 4096):
        end = min(start + 4097, len(ids))
        chunk = ids[start:end]
        if chunk.numel() < 2:
            continue

        # Score y = chunk[1:] from x = chunk[:-1], clipped naturally
        # by BLOCK in model.forward.
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        logits = model(x)
        # logits may only represent the final BLOCK context when x > BLOCK.
        # To avoid ambiguity, score in BLOCK-sized windows.
        for j in range(0, x.shape[1], BLOCK):
            xx = x[:, j:j + BLOCK]
            yy = y[:, j:j + BLOCK]
            if yy.numel() == 0:
                continue
            ll = model(xx)
            loss_sum = F.cross_entropy(
                ll.reshape(-1, 98).float(),
                yy.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss_sum.item())
            total_n += yy.numel()

    return total_nll / total_n


# ============================================================
# Teacher-forced loss/accuracy on selected validation anchors
# ============================================================

def make_validation_anchors(count=7, continuation=GENERATE_CHARS + 1):
    # Deterministic spread over the held-out split.
    max_start = len(val_text) - continuation - 2
    if max_start <= BLOCK:
        raise RuntimeError("Validation split too small for anchor selection.")

    positions = np.linspace(
        BLOCK,
        max_start,
        count,
        dtype=np.int64,
    )

    anchors = []
    for i, pos in enumerate(positions):
        context_start = max(0, int(pos) - 32)
        prompt = val_text[context_start:int(pos)]
        target = val_text[int(pos):int(pos) + continuation]
        anchors.append((f"val_anchor_{i+1}", prompt, target))
    return anchors


ANCHORS = make_validation_anchors()


@torch.inference_mode()
def teacher_forced_anchor_metrics(model, prompt, target_text):
    full = encode(prompt + target_text)
    prompt_len = len(prompt)

    losses = []
    top1 = []
    top5 = []

    # Batch in windows. Each target gets its true previous context.
    # We evaluate targets one BLOCK context at a time.
    target_ids = encode(target_text)

    prev = encode(prompt)
    combined = torch.cat([prev, target_ids], dim=0)
    target_start = len(prev)

    for s in range(target_start, len(combined), BLOCK):
        e = min(s + BLOCK, len(combined))
        # Include one preceding token for the first target in this block.
        left = max(0, s - BLOCK)
        x = combined[left:e - 0].unsqueeze(0)
        # Target positions correspond to x positions offset by one.
        # We need logits for positions whose next chars are the current
        # targets in [s:e).
        if x.shape[1] < 2:
            continue

        y_start = max(s, left + 1)
        xx = combined[left:y_start - 1 if y_start - 1 > left else e - 1]
        # Simpler robust per-window construction:
        xx = combined[max(0, s - BLOCK):e - 1]
        yy = combined[s:e]

        if xx.numel() == 0 or yy.numel() == 0:
            continue

        logits = model(xx.unsqueeze(0))[0, -yy.numel():]

        lp = F.log_softmax(logits.float(), dim=-1)
        y = yy

        losses.extend((-lp[torch.arange(len(y), device=DEVICE), y]).tolist())
        top1.extend(
            (torch.argmax(lp, dim=-1) == y).float().tolist()
        )

        k = min(5, lp.shape[-1])
        top5_idx = torch.topk(lp, k=k, dim=-1).indices
        top5.extend(
            (top5_idx == y.unsqueeze(-1)).any(dim=-1).float().tolist()
        )

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "top5": float(np.mean(top5)) if top5 else float("nan"),
    }


# ============================================================
# Sampling / generation
# ============================================================

def choose_next(logits, mode, temperature=1.0):
    logits = logits.float()
    if mode == "greedy":
        return torch.argmax(logits)

    logits = logits / temperature
    k = min(TOP_K, logits.numel())
    values, indices = torch.topk(logits, k=k)
    probs = F.softmax(values, dim=-1)
    idx = torch.multinomial(probs, 1).item()
    return indices[idx]


REFERENCE_WORDS = set()
w = []
for ch in text:
    if ch.isalpha() or ch == "'":
        w.append(ch)
    elif w:
        REFERENCE_WORDS.add("".join(w).lower())
        w = []
if w:
    REFERENCE_WORDS.add("".join(w).lower())


def text_metrics(s):
    words = s.split()
    cleaned = []

    for word in words:
        token = word.strip(".,!?;:\"'()[]{}<>“”‘’—-")
        token = "".join(c for c in token if c.isalpha() or c == "'")
        if token:
            cleaned.append(token.lower())

    total = len(cleaned)
    valid = sum(w in REFERENCE_WORDS for w in cleaned)

    counts = Counter(cleaned)
    repeated = sum(c - 1 for c in counts.values() if c > 1)

    return {
        "words": total,
        "valid_word_ratio": valid / total if total else 0.0,
        "malformed_word_ratio": 1.0 - (valid / total) if total else 0.0,
        "avg_word_len": (
            sum(len(w) for w in cleaned) / total if total else 0.0
        ),
        "vocab_diversity": (
            len(counts) / total if total else 0.0
        ),
        "punctuation_ratio": (
            sum(c in ".,!?;:\"'()[]{}-—" for c in s) / len(s)
            if s else 0.0
        ),
        "whitespace_ratio": (
            sum(c.isspace() for c in s) / len(s) if s else 0.0
        ),
        "repeated_word_ratio": (
            repeated / total if total else 0.0
        ),
    }


@torch.inference_mode()
def generate(model, prompt, mode, temperature, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    ids = encode(prompt)

    generated = []

    for _ in range(GENERATE_CHARS):
        x = ids[-BLOCK:].unsqueeze(0)
        logits = model(x)[0, -1]
        token = choose_next(logits, mode, temperature)
        ids = torch.cat([ids, token.view(1)])
        generated.append(int(token.item()))

    return decode(generated)


# ============================================================
# Free-running drift against REAL validation anchors
# ============================================================

@torch.inference_mode()
def free_running_drift(model, prompt, target_text, mode, temperature, seed):
    # We feed the prompt, then our own predictions. At each horizon,
    # ask: "Given the model's self-generated history so far, how well
    # does it predict the actual held-out next character?"
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    ids = encode(prompt)
    target = encode(target_text)
    if target.numel() < GENERATE_CHARS + 1:
        raise RuntimeError(
            f'Free-running target must contain at least {GENERATE_CHARS + 1} '
            f'characters, found {target.numel()}.'
        )

    out = []

    for step in range(GENERATE_CHARS):
        h = step if step < len(target) else len(target) - 1

        logits = model(ids[-BLOCK:].unsqueeze(0))[0, -1].float()
        real_target = target[step]

        logp = F.log_softmax(logits, dim=-1)
        nll = float((-logp[real_target]).item())
        top1 = int(torch.argmax(logp).item() == int(real_target.item()))

        token = choose_next(logits, mode, temperature)
        ids = torch.cat([ids, token.view(1)])
        out.append(int(token.item()))

        out.append if False else None  # no-op for clarity

    # Recompute checkpoint horizons from the same generated continuation.
    generated = torch.tensor(out, dtype=torch.long, device=DEVICE)
    results = {}

    for horizon in HORIZONS:
        h = min(horizon, GENERATE_CHARS)

        context = torch.cat([encode(prompt), generated[:h]])
        logits = model(context[-BLOCK:].unsqueeze(0))[0, -1].float()

        target_id = target[h]
        logp = F.log_softmax(logits, dim=-1)

        results[h] = {
            "gt_nll": float((-logp[target_id]).item()),
            "gt_top1": int(torch.argmax(logp).item() == int(target_id.item())),
            "text_metrics": text_metrics(
                decode(generated[:h].tolist())
            ),
        }

    return results, decode(generated.tolist())


# ============================================================
# Main
# ============================================================

loaded = {}
for name, path in MODELS.items():
    print(f"Loading {name}: {path}")
    model, step, best_val = load_model(path)
    loaded[name] = (model, step, best_val)
    print(f"  step={step} best_val={best_val}")


report = []
rows = []

report.append("=" * 72)
report.append("GLYPH V2 GENERATION DYNAMICS BENCHMARK — FIXED")
report.append("=" * 72)
report.append("Fixed prompts are for qualitative generation only.")
report.append(
    "Free-running drift uses real held-out validation anchors, so the "
    "ground-truth continuation is valid and actually belongs to the "
    "250k held-out split."
)
report.append("")

# Teacher-forced anchor benchmark.
report.append("=" * 72)
report.append("TEACHER-FORCED HELD-OUT ANCHORS")
report.append("=" * 72)

for name, (model, step, best_val) in loaded.items():
    losses = []
    top1s = []
    top5s = []

    for anchor_name, prompt, target in ANCHORS:
        m = teacher_forced_anchor_metrics(model, prompt, target)
        losses.append(m["loss"])
        top1s.append(m["top1"])
        top5s.append(m["top5"])

    report.append(
        f"{name:18s} step={step!s:>6s} "
        f"loss={np.mean(losses):.6f} "
        f"ppl={math.exp(np.mean(losses)):.3f} "
        f"top1={np.mean(top1s):.4f} "
        f"top5={np.mean(top5s):.4f}"
    )

# Fixed-prompt generation.
for mi, (name, (model, step, best_val)) in enumerate(loaded.items()):
    for mode, temperatures in [
        ("greedy", [1.0]),
        ("sample", TEMPS),
    ]:
        for temp in temperatures:
            label = "greedy" if mode == "greedy" else f"topk{TOP_K}_t{temp:.1f}"
            report.append("")
            report.append("=" * 72)
            report.append(f"MODEL: {name} | MODE: {label}")
            report.append("=" * 72)

            for pi, prompt in enumerate(FIXED_PROMPTS):
                t0 = time.perf_counter()
                continuation = generate(
                    model,
                    prompt,
                    mode,
                    temp,
                    SEED + mi * 1000 + pi,
                )
                elapsed = time.perf_counter() - t0

                m = text_metrics(continuation)
                report.append("")
                report.append(f"PROMPT: {prompt!r}")
                report.append(continuation)
                report.append(
                    f"FULL500 valid={m['valid_word_ratio']:.4f} "
                    f"malformed={m['malformed_word_ratio']:.4f} "
                    f"avg_word_len={m['avg_word_len']:.4f} "
                    f"diversity={m['vocab_diversity']:.4f} "
                    f"punct={m['punctuation_ratio']:.4f} "
                    f"space={m['whitespace_ratio']:.4f} "
                    f"repeat_word={m['repeated_word_ratio']:.4f} "
                    f"time={elapsed:.3f}s"
                )

# Free-running drift using held-out anchors.
for mi, (name, (model, step, best_val)) in enumerate(loaded.items()):
    report.append("")
    report.append("=" * 72)
    report.append(f"FREE-RUNNING DRIFT: {name}")
    report.append("=" * 72)

    for mode, temperatures in [
        ("greedy", [1.0]),
        ("sample", TEMPS),
    ]:
        for temp in temperatures:
            label = "greedy" if mode == "greedy" else f"topk{TOP_K}_t{temp:.1f}"
            report.append("")
            report.append(f"MODE: {label}")

            for ai, (anchor_name, prompt, target) in enumerate(ANCHORS):
                t0 = time.perf_counter()
                drift, continuation = free_running_drift(
                    model,
                    prompt,
                    target,
                    mode,
                    temp,
                    SEED + mi * 1000 + ai,
                )
                elapsed = time.perf_counter() - t0

                report.append("")
                report.append(f"{anchor_name}: prompt={prompt!r}")
                report.append(f"GENERATED: {continuation[:200]}...")

                for h in HORIZONS:
                    r = drift[h]
                    tm = r["text_metrics"]
                    report.append(
                        f"H{h:>3d}: gt_nll={r['gt_nll']:.5f} "
                        f"gt_top1={r['gt_top1']} "
                        f"valid={tm['valid_word_ratio']:.4f} "
                        f"malformed={tm['malformed_word_ratio']:.4f}"
                    )

                    rows.append({
                        "model": name,
                        "step": step,
                        "best_val_loss": best_val,
                        "mode": label,
                        "anchor": anchor_name,
                        "prompt": prompt,
                        "horizon": h,
                        "gt_nll": r["gt_nll"],
                        "gt_top1": r["gt_top1"],
                        "valid_word_ratio": tm["valid_word_ratio"],
                        "malformed_word_ratio": tm["malformed_word_ratio"],
                    })

                report.append(f"time={elapsed:.3f}s")


# Aggregate summary.
report.append("")
report.append("=" * 72)
report.append("AGGREGATE FREE-RUNNING DRIFT")
report.append("=" * 72)
report.append(
    "Lower gt_nll / higher gt_top1 after self-generated history means "
    "the model stays better aligned with real held-out text."
)

for name in MODELS:
    for label in ["greedy"] + [f"topk{TOP_K}_t{t:.1f}" for t in TEMPS]:
        subset = [r for r in rows if r["model"] == name and r["mode"] == label]
        report.append("")
        report.append(f"{name} | {label}")
        for h in HORIZONS:
            hs = [r for r in subset if r["horizon"] == h]
            if not hs:
                continue
            report.append(
                f"H{h:>3d}: "
                f"gt_nll={np.mean([r['gt_nll'] for r in hs]):.5f} "
                f"gt_top1={np.mean([r['gt_top1'] for r in hs]):.4f} "
                f"valid={np.mean([r['valid_word_ratio'] for r in hs]):.4f} "
                f"malformed={np.mean([r['malformed_word_ratio'] for r in hs]):.4f}"
            )


with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model", "step", "best_val_loss", "mode", "anchor",
            "prompt", "horizon", "gt_nll", "gt_top1",
            "valid_word_ratio", "malformed_word_ratio",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 72)
print("BENCHMARK COMPLETE")
print(f"Text report : {OUT_TXT}")
print(f"CSV report  : {OUT_CSV}")
print("=" * 72)
