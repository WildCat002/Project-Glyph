import csv
import math
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
MAX_DATA = 5_000_000
VAL_CHARS = 250_000
TRAIN_END = MAX_DATA - VAL_CHARS

BLOCK = 128
N_LAYER = 4
N_HEAD = 4
N_EMB = 128
VOCAB = 98

ANCHOR_COUNT = 7
ANCHOR_CONTEXT = 64
GENERATE_CHARS = 500
HORIZONS = [10, 25, 50, 100, 250, 500]

TOP_K = 20
TEMPERATURES = [0.7, 0.9, 1.0]

PROMPTS = [
    "alice was",
    "the rabbit",
    "once upon a time",
    "the king",
    "she looked",
    "the man",
    "what",
]

MODELS = {
    "Glyph v2 20k": "glyph_v2.pt",
    "Glyph v2 50k": "glyph_v2_50k_best.pt",
    "Glyph v2 100k": "glyph_v2_100k_cuda_best.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_TXT = "glyph_v2_generation_dynamics_v2_results.txt"
OUT_CSV = "glyph_v2_generation_dynamics_v2_results.csv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("=" * 76)
print("GLYPH V2 GENERATION DYNAMICS v2")
print("=" * 76)
print(f"Device       : {DEVICE}")
print(f"PyTorch      : {torch.__version__}")
if DEVICE.type == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 76)

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()[:MAX_DATA].lower()

chars = sorted(set(text))
if len(chars) != VOCAB:
    raise RuntimeError(f"Expected vocab={VOCAB}, found {len(chars)}")
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
val_text = text[TRAIN_END:MAX_DATA]

print(f"Total chars  : {len(text):,}")
print(f"Train chars  : {TRAIN_END:,}")
print(f"Val chars    : {len(val_text):,}")
print(f"Vocab        : {len(chars)}")


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
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
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
        self.tok = nn.Embedding(VOCAB, N_EMB)
        self.pos = nn.Embedding(BLOCK, N_EMB)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMB)
        self.head = nn.Linear(N_EMB, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        _, t = idx.shape
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
        raise RuntimeError(f"{path}: missing model state")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt.get("step", "?"), ckpt.get("best_val_loss")


def encode(s):
    bad = [c for c in s if c not in stoi]
    if bad:
        raise ValueError(f"Unknown chars: {bad!r}")
    return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=DEVICE)


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


# Deterministic held-out anchors. These are real validation contexts, not the fixed prompts.
if len(val_text) < ANCHOR_CONTEXT + GENERATE_CHARS + 1:
    raise RuntimeError("Validation split is too small.")
positions = np.linspace(
    ANCHOR_CONTEXT,
    len(val_text) - GENERATE_CHARS - 1,
    ANCHOR_COUNT,
    dtype=np.int64,
)
anchors = []
for i, pos in enumerate(positions):
    pos = int(pos)
    anchors.append({
        "name": f"val_anchor_{i + 1}",
        "context": val_text[pos - ANCHOR_CONTEXT:pos],
        "target": val_text[pos:pos + GENERATE_CHARS],
    })

print("Held-out anchors:")
for a in anchors:
    print(f"  {a['name']}: {a['context']!r}")

# Reference word vocabulary for descriptive generation diagnostics.
reference_words = set()
w = []
for ch in text:
    if ch.isalpha() or ch == "'":
        w.append(ch)
    elif w:
        reference_words.add("".join(w).lower())
        w = []
if w:
    reference_words.add("".join(w).lower())


def text_metrics(s):
    words = s.split()
    cleaned = []
    for token in words:
        token = token.strip(".,!?;:\"'()[]{}<>“”‘’—-")
        token = "".join(c for c in token if c.isalpha() or c == "'")
        if token:
            cleaned.append(token.lower())
    total = len(cleaned)
    if not total:
        return {
            "valid_word_ratio": 0.0,
            "malformed_word_ratio": 0.0,
            "avg_word_len": 0.0,
            "vocab_diversity": 0.0,
            "repeated_word_ratio": 0.0,
            "punctuation_ratio": sum(c in ".,!?;:\"'()[]{}-—" for c in s) / len(s) if s else 0.0,
            "whitespace_ratio": sum(c.isspace() for c in s) / len(s) if s else 0.0,
        }
    counts = Counter(cleaned)
    valid = sum(w in reference_words for w in cleaned)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "valid_word_ratio": valid / total,
        "malformed_word_ratio": 1.0 - valid / total,
        "avg_word_len": sum(map(len, cleaned)) / total,
        "vocab_diversity": len(counts) / total,
        "repeated_word_ratio": repeated / total,
        "punctuation_ratio": sum(c in ".,!?;:\"'()[]{}-—" for c in s) / len(s) if s else 0.0,
        "whitespace_ratio": sum(c.isspace() for c in s) / len(s) if s else 0.0,
    }


def choose_next(logits, mode, temperature):
    logits = logits.float()
    if mode == "greedy":
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    k = min(TOP_K, logits.numel())
    values, indices = torch.topk(logits, k=k)
    probs = F.softmax(values, dim=-1)
    return indices[torch.multinomial(probs, num_samples=1).squeeze(-1)]


@torch.inference_mode()
def run_anchor(model, context_ids, target_ids, mode, temperature, seed):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    current = context_ids.clone()
    nlls = []
    hits = []
    generated = []

    for t in range(GENERATE_CHARS):
        logits = model(current[-BLOCK:].unsqueeze(0))[0, -1].float()
        real_target = target_ids[t]
        logp = F.log_softmax(logits, dim=-1)
        nlls.append(float(-logp[real_target].item()))
        hits.append(int(torch.argmax(logp).item() == int(real_target.item())))
        token = choose_next(logits, mode, temperature)
        generated.append(int(token.item()))
        current = torch.cat([current, token.view(1)])

    gen_ids = torch.tensor(generated, dtype=torch.long, device=DEVICE)
    out = {}
    for h in HORIZONS:
        s = decode(gen_ids[:h].tolist())
        tm = text_metrics(s)
        out[h] = {
            "avg_gt_nll": float(np.mean(nlls[:h])),
            "avg_gt_top1": float(np.mean(hits[:h])),
            **tm,
        }
    return out, decode(gen_ids.tolist())


@torch.inference_mode()
def generate_fixed_prompt(model, prompt, mode, temperature, seed):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    current = encode(prompt)
    out = []
    for _ in range(GENERATE_CHARS):
        logits = model(current[-BLOCK:].unsqueeze(0))[0, -1].float()
        token = choose_next(logits, mode, temperature)
        out.append(int(token.item()))
        current = torch.cat([current, token.view(1)])
    return decode(out)


loaded = {}
for name, path in MODELS.items():
    print(f"Loading {name}: {path}")
    model, step, best_val = load_model(path)
    loaded[name] = {"model": model, "step": step, "best_val_loss": best_val}
    print(f"  step={step} best_val={best_val}")

report = [
    "=" * 76,
    "GLYPH V2 GENERATION DYNAMICS v2",
    "=" * 76,
    "Each H value is the AVERAGE over ALL generated positions 1..H.",
    "Ground-truth targets come from real held-out validation anchors.",
    "",
]
rows = []

for mi, (name, info) in enumerate(loaded.items()):
    for mode, temps in [("greedy", [1.0]), ("sample", TEMPERATURES)]:
        for temp in temps:
            label = "greedy" if mode == "greedy" else f"topk{TOP_K}_t{temp:.1f}"
            print(f"Running {name} | {label}")
            ag = {h: {k: [] for k in ["nll", "top1", "valid", "malformed", "repeat"]} for h in HORIZONS}

            for ai, anchor in enumerate(anchors):
                t0 = time.perf_counter()
                result, generated = run_anchor(
                    info["model"],
                    encode(anchor["context"]),
                    encode(anchor["target"]),
                    mode,
                    temp,
                    SEED + mi * 1000 + ai * 100 + int(temp * 10),
                )
                elapsed = time.perf_counter() - t0

                report += [
                    "",
                    f"{name} | {label} | {anchor['name']}",
                    f"CONTEXT: {anchor['context']!r}",
                    f"GENERATED: {generated}",
                ]

                for h in HORIZONS:
                    r = result[h]
                    ag[h]["nll"].append(r["avg_gt_nll"])
                    ag[h]["top1"].append(r["avg_gt_top1"])
                    ag[h]["valid"].append(r["valid_word_ratio"])
                    ag[h]["malformed"].append(r["malformed_word_ratio"])
                    ag[h]["repeat"].append(r["repeated_word_ratio"])
                    rows.append({
                        "model": name,
                        "step": info["step"],
                        "best_val_loss": info["best_val_loss"],
                        "mode": label,
                        "anchor": anchor["name"],
                        "horizon": h,
                        "avg_gt_nll": r["avg_gt_nll"],
                        "avg_gt_top1": r["avg_gt_top1"],
                        "valid_word_ratio": r["valid_word_ratio"],
                        "malformed_word_ratio": r["malformed_word_ratio"],
                        "avg_word_len": r["avg_word_len"],
                        "vocab_diversity": r["vocab_diversity"],
                        "repeated_word_ratio": r["repeated_word_ratio"],
                    })
                    report.append(
                        f"H{h:>3d}: avg_gt_nll={r['avg_gt_nll']:.5f} "
                        f"avg_gt_top1={r['avg_gt_top1']:.4f} "
                        f"valid={r['valid_word_ratio']:.4f} "
                        f"malformed={r['malformed_word_ratio']:.4f}"
                    )
                report.append(f"time={elapsed:.3f}s")

            report.append("")
            report.append(f"AGGREGATE: {name} | {label}")
            for h in HORIZONS:
                report.append(
                    f"H{h:>3d}: avg_gt_nll={np.mean(ag[h]['nll']):.5f} "
                    f"avg_gt_top1={np.mean(ag[h]['top1']):.4f} "
                    f"valid={np.mean(ag[h]['valid']):.4f} "
                    f"malformed={np.mean(ag[h]['malformed']):.4f} "
                    f"repeat={np.mean(ag[h]['repeat']):.4f}"
                )

report += ["", "=" * 76, "FIXED-PROMPT QUALITATIVE SAMPLES", "=" * 76]
report.append("Fixed prompts are qualitative only; they are not used for drift scoring.")

for mi, (name, info) in enumerate(loaded.items()):
    for mode, temps in [("greedy", [1.0]), ("sample", TEMPERATURES)]:
        for temp in temps:
            label = "greedy" if mode == "greedy" else f"topk{TOP_K}_t{temp:.1f}"
            report += ["", f"MODEL: {name} | {label}"]
            for pi, prompt in enumerate(PROMPTS):
                t0 = time.perf_counter()
                generated = generate_fixed_prompt(
                    info["model"], prompt, mode, temp,
                    SEED + 50000 + mi * 1000 + pi * 10 + int(temp * 10),
                )
                elapsed = time.perf_counter() - t0
                tm = text_metrics(generated)
                report += [
                    "",
                    f"PROMPT: {prompt!r}",
                    generated,
                    f"FULL500: valid={tm['valid_word_ratio']:.4f} "
                    f"malformed={tm['malformed_word_ratio']:.4f} "
                    f"avg_word_len={tm['avg_word_len']:.4f} "
                    f"diversity={tm['vocab_diversity']:.4f} "
                    f"repeat={tm['repeated_word_ratio']:.4f} "
                    f"time={elapsed:.3f}s",
                ]

report += [
    "",
    "=" * 76,
    "INTERPRETATION",
    "=" * 76,
    "avg_gt_nll = average NLL of the REAL held-out character under the model's OWN generated history.",
    "avg_gt_top1 = average top-1 accuracy for the REAL held-out character under that self-generated history.",
    "H10/H25/... are cumulative averages across all positions 1..H, not single-position snapshots.",
    "Rising NLL / falling top-1 with H indicates growing divergence from the held-out text distribution.",
    "Word metrics are descriptive diagnostics, not universal English-quality scores.",
]

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    fields = [
        "model", "step", "best_val_loss", "mode", "anchor", "horizon",
        "avg_gt_nll", "avg_gt_top1", "valid_word_ratio",
        "malformed_word_ratio", "avg_word_len", "vocab_diversity",
        "repeated_word_ratio",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("=" * 76)
print("BENCHMARK COMPLETE")
print(f"Text report : {OUT_TXT}")
print(f"CSV report  : {OUT_CSV}")
print("=" * 76)
