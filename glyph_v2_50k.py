"""
Glyph v2 Extended Training
==========================

Continue the EXISTING Glyph v2 20,000-step checkpoint to step 50,000.

This is intentionally NOT a new architecture.

The experiment keeps:
    BLOCK       = 128
    N_LAYER     = 4
    N_HEAD      = 4
    N_EMB       = 128
    BATCH       = 32
    WD          = 0.01
    GRAD_CLIP   = 0.05
    ADAM_EPS    = 1e-6

It resumes the exact model + AdamW optimizer state from:
    glyph_v2.pt

At step 20,000 the original v2 schedule had reached:
    LR = 1e-6

For the extension, we keep LR fixed at 1e-6. This avoids an abrupt LR
increase and makes this experiment specifically about giving the SAME
trained model another 30,000 optimization steps.

Outputs are kept separate from the original v2 files:
    glyph_v2_50k.pt
    glyph_v2_50k_best.pt
    glyph_v2_50k_last_good.pt
    glyph_v2_50k_history.csv

The script automatically resumes glyph_v2_50k.pt when a previous continuation
checkpoint exists between 20,000 and 50,000 steps, so interrupted progress is
not discarded.

The CPU thread count is set to 2 because the local speed benchmark measured
2 threads as the fastest configuration on the current 4-logical-CPU laptop.
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# REPRODUCIBILITY / CPU
# ============================================================

SEED = 42
CPU_THREADS = 2

torch.set_num_threads(CPU_THREADS)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# FILES
# ============================================================

DATA_FILE = "data.txt"

BASE_CKPT = "glyph_v2.pt"

CKPT = "glyph_v2_50k.pt"
BEST_CKPT = "glyph_v2_50k_best.pt"
LAST_GOOD_CKPT = "glyph_v2_50k_last_good.pt"
HISTORY_FILE = "glyph_v2_50k_history.csv"


# ============================================================
# DATA
# ============================================================

MAX_DATA = 5_000_000
VAL_CHARS = 250_000
TRAIN_END = MAX_DATA - VAL_CHARS

raw_text = open(
    DATA_FILE,
    "r",
    encoding="utf-8",
    errors="ignore",
).read().lower()

text = raw_text[:MAX_DATA]

if len(text) < MAX_DATA:
    print(
        f"Warning: data.txt contains only {len(text):,} chars; "
        f"expected {MAX_DATA:,}."
    )

if len(text) <= VAL_CHARS + 1:
    raise ValueError(
        "Not enough text for the requested train/validation split."
    )

train_text = text[:TRAIN_END]
val_text = text[TRAIN_END:]

chars = sorted(set(text))

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

V = len(chars)

train_data = np.array(
    [stoi[c] for c in train_text],
    dtype=np.uint16,
)

val_data = np.array(
    [stoi[c] for c in val_text],
    dtype=np.uint16,
)

print("=" * 64)
print("Glyph v2 Extended Training")
print("=" * 64)
print("Batching     : vectorized PyTorch indexing")
print("Validation   : every 1,000 steps / 64 batches")
print("Checkpoint   : every 2,000 steps + every new best")
print("=" * 64)
print(f"Total chars : {len(text):,}")
print(f"Train chars : {len(train_data):,}")
print(f"Val chars   : {len(val_data):,}")
print(f"Vocab       : {V}")
print()


# ============================================================
# ARCHITECTURE
# ============================================================

BLOCK = 128

N_LAYER = 4
N_HEAD = 4
N_EMB = 128

DROPOUT = 0.1


# ============================================================
# TRAINING
# ============================================================

BATCH = 32

SOURCE_STEP_REQUIRED = 20_000
TARGET_STEP = 50_000

# This is the LR reached by the original v2 schedule at step 20,000.
CONTINUATION_LR = 1e-6

WD = 0.01
GRAD_CLIP = 0.05
ADAM_EPS = 1e-6

SAVE_EVERY = 2_000
EVAL_EVERY = 1_000

VAL_BATCHES = 64


# ============================================================
# MODEL
# ============================================================


class Attn(nn.Module):
    def __init__(self):
        super().__init__()

        self.qkv = nn.Linear(
            N_EMB,
            3 * N_EMB,
            bias=False,
        )

        self.proj = nn.Linear(
            N_EMB,
            N_EMB,
            bias=False,
        )

        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(N_EMB, dim=2)

        head_dim = C // N_HEAD

        q = (
            q.view(
                B,
                T,
                N_HEAD,
                head_dim,
            )
            .transpose(1, 2)
            .float()
        )

        k = (
            k.view(
                B,
                T,
                N_HEAD,
                head_dim,
            )
            .transpose(1, 2)
            .float()
        )

        v = (
            v.view(
                B,
                T,
                N_HEAD,
                head_dim,
            )
            .transpose(1, 2)
            .float()
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.drop(
            self.proj(y)
        )


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(N_EMB)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(N_EMB)

        self.mlp = nn.Sequential(
            nn.Linear(
                N_EMB,
                4 * N_EMB,
            ),
            nn.GELU(),
            nn.Linear(
                4 * N_EMB,
                N_EMB,
            ),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(
            self.ln1(x)
        )
        x = x + self.mlp(
            self.ln2(x)
        )
        return x


class Glyph(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok = nn.Embedding(
            V,
            N_EMB,
        )

        self.pos = nn.Embedding(
            BLOCK,
            N_EMB,
        )

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(N_LAYER)
            ]
        )

        self.ln_f = nn.LayerNorm(
            N_EMB
        )

        self.head = nn.Linear(
            N_EMB,
            V,
            bias=False,
        )

        self._initialize_weights()

        # Weight tying, same as original v2.
        self.head.weight = self.tok.weight

        # Same GPT-style residual projection scaling as original v2.
        scale = 1.0 / math.sqrt(
            2 * N_LAYER
        )

        for block in self.blocks:
            nn.init.normal_(
                block.attn.proj.weight,
                mean=0.0,
                std=0.02 * scale,
            )

            nn.init.normal_(
                block.mlp[2].weight,
                mean=0.0,
                std=0.02 * scale,
            )

    def _initialize_weights(self):
        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Embedding,
            ):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(
        self,
        idx,
        targets=None,
    ):
        B, T = idx.shape

        if T > BLOCK:
            raise ValueError(
                f"Sequence length {T} exceeds BLOCK={BLOCK}"
            )

        pos = torch.arange(
            T,
            device=idx.device,
        )

        x = (
            self.tok(idx)
            + self.pos(pos)
        )

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.float().reshape(-1, V),
            targets.reshape(-1),
        )

        return logits, loss


# ============================================================
# MODEL / OPTIMIZER
# ============================================================

device = torch.device("cpu")

model = Glyph().to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters  : {num_params:,} "
    f"({num_params / 1e6:.2f}M)"
)
print(f"Context     : {BLOCK}")
print(f"Layers      : {N_LAYER}")
print(f"Heads       : {N_HEAD}")
print(f"Embedding   : {N_EMB}")
print()


decay = []
no_decay = []

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    if param.dim() >= 2:
        decay.append(param)
    else:
        no_decay.append(param)

print(
    "Decay params    : "
    f"{sum(p.numel() for p in decay):,}"
)

print(
    "No-decay params : "
    f"{sum(p.numel() for p in no_decay):,}"
)

opt = torch.optim.AdamW(
    [
        {
            "params": decay,
            "weight_decay": WD,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ],
    lr=CONTINUATION_LR,
    betas=(0.9, 0.95),
    eps=ADAM_EPS,
)


# ============================================================
# BATCHES
# ============================================================


# Convert the corpus once. The old implementation rebuilt each batch
# with Python list-comprehensions + np.stack(). This version performs the
# same random window selection, but the actual batch extraction is handled
# by PyTorch indexing.
train_tensor = torch.from_numpy(train_data)
val_tensor = torch.from_numpy(val_data)

_batch_offsets = torch.arange(
    BLOCK + 1,
    dtype=torch.long,
)


def get_batch(
    source,
    bs,
):
    if source.numel() <= BLOCK + 1:
        raise ValueError(
            "Dataset split is too small for the context length."
        )

    # Keep NumPy's RNG here so the random window-selection mechanism remains
    # the same as the original script.
    ix_np = np.random.randint(
        0,
        source.numel() - BLOCK - 1,
        size=bs,
    )

    ix = torch.from_numpy(ix_np).long()

    # Shape: [BATCH, BLOCK + 1]
    positions = (
        ix[:, None]
        + _batch_offsets[None, :]
    )

    batch = source[positions].long()

    return (
        batch[:, :-1],
        batch[:, 1:],
    )


# ============================================================
# FINITE CHECKS
# ============================================================


def parameters_are_finite():
    return all(
        torch.isfinite(param).all().item()
        for param in model.parameters()
    )


def gradients_are_finite():
    for param in model.parameters():

        if param.grad is None:
            continue

        if not torch.isfinite(
            param.grad
        ).all():
            return False

    return True


def max_weight():
    values = []

    for param in model.parameters():
        values.append(
            float(
                param.detach()
                .abs()
                .max()
            )
        )

    return max(values) if values else 0.0


# ============================================================
# VALIDATION
# ============================================================


@torch.no_grad()
def evaluate_validation(
    num_batches=VAL_BATCHES,
):
    model.eval()

    total_loss = 0.0

    for _ in range(num_batches):

        x, y = get_batch(
            val_tensor,
            BATCH,
        )

        _, loss = model(
            x,
            y,
        )

        if not torch.isfinite(loss):
            model.train()
            return float("inf")

        total_loss += loss.item()

    model.train()

    return total_loss / num_batches


# ============================================================
# CHECKPOINTS
# ============================================================


def make_state(
    step,
    best_val_loss,
):
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {
            "run": "glyph_v2_extended_50k",
            "base_checkpoint": BASE_CKPT,
            "source_step": SOURCE_STEP_REQUIRED,
            "eval_every": EVAL_EVERY,
            "val_batches": VAL_BATCHES,
            "save_every": SAVE_EVERY,
            "batching": "vectorized_torch_indexing",
            "target_step": TARGET_STEP,
            "continuation_lr": CONTINUATION_LR,
            "block": BLOCK,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_emb": N_EMB,
            "batch": BATCH,
            "weight_decay": WD,
            "grad_clip": GRAD_CLIP,
            "adam_eps": ADAM_EPS,
            "train_chars": len(train_data),
            "val_chars": len(val_data),
            "vocab": V,
        },
    }


def atomic_torch_save(
    state,
    path,
):
    tmp = f"{path}.tmp"

    torch.save(
        state,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def save_checkpoint(
    step,
    best_val_loss,
    best=False,
):
    state = make_state(
        step,
        best_val_loss,
    )

    atomic_torch_save(
        state,
        CKPT,
    )

    atomic_torch_save(
        state,
        LAST_GOOD_CKPT,
    )

    if best:
        atomic_torch_save(
            state,
            BEST_CKPT,
        )


# ============================================================
# LOAD / RESUME CHECKPOINT
# ============================================================

# If a previous 50k continuation was interrupted after making progress,
# resume it. Otherwise start from the original 20k V2 checkpoint.
resume_path = None

if os.path.exists(CKPT):
    try:
        probe = torch.load(
            CKPT,
            map_location="cpu",
            weights_only=False,
        )
        probe_step = int(
            probe.get("step", 0)
        )

        if SOURCE_STEP_REQUIRED <= probe_step < TARGET_STEP:
            resume_path = CKPT
    except Exception:
        resume_path = None

checkpoint_path = (
    resume_path
    if resume_path is not None
    else BASE_CKPT
)

if not os.path.exists(
    checkpoint_path
):
    raise FileNotFoundError(
        f"Missing checkpoint: {checkpoint_path}. "
        "Run the original Glyph v2 training first."
    )

print(
    f"Loading checkpoint: "
    f"{checkpoint_path}"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

source_step = int(
    checkpoint.get("step", 0)
)

if not (
    SOURCE_STEP_REQUIRED
    <= source_step
    < TARGET_STEP
):
    raise RuntimeError(
        f"Expected checkpoint step in "
        f"[{SOURCE_STEP_REQUIRED:,}, {TARGET_STEP:,}), "
        f"but found {source_step:,}."
    )

source_config = checkpoint.get(
    "config",
    {}
)

# The original V2 checkpoint has the canonical architecture/training config.
# The continuation checkpoint has the same architecture plus its continuation
# metadata. In both cases these fields must remain identical.
expected = {
    "block": BLOCK,
    "n_layer": N_LAYER,
    "n_head": N_HEAD,
    "n_emb": N_EMB,
    "batch": BATCH,
    "weight_decay": WD,
    "grad_clip": GRAD_CLIP,
    "adam_eps": ADAM_EPS,
    "train_chars": len(train_data),
    "val_chars": len(val_data),
    "vocab": V,
}

for key, expected_value in expected.items():

    actual_value = source_config.get(
        key
    )

    if actual_value != expected_value:
        raise RuntimeError(
            f"Checkpoint config mismatch for "
            f"{key}: expected {expected_value!r}, "
            f"found {actual_value!r}."
        )

model.load_state_dict(
    checkpoint["model"]
)

opt.load_state_dict(
    checkpoint["opt"]
)

# Force the continuation LR explicitly.
for group in opt.param_groups:
    group["lr"] = CONTINUATION_LR

start_step = source_step

best_val_loss = float(
    checkpoint.get(
        "best_val_loss",
        float("inf"),
    )
)

print(
    f"Loaded step     : {start_step:,}"
)

print(
    f"Best val loss   : "
    f"{best_val_loss:.6f}"
)

print(
    f"Continuation LR : "
    f"{CONTINUATION_LR:.2e}"
)

print(
    f"CPU threads     : "
    f"{CPU_THREADS}"
)

print()

# SAFETY EVALUATION BEFORE CONTINUING
# ============================================================

if not parameters_are_finite():
    raise RuntimeError(
        "Source checkpoint contains "
        "non-finite model parameters."
    )

anchor_val = evaluate_validation()

print(
    f"Validation at step {start_step:,}: "
    f"{anchor_val:.6f}"
)

if not math.isfinite(anchor_val):
    raise RuntimeError(
        "Source checkpoint produced "
        "non-finite validation loss."
    )

print()


# ============================================================
# HISTORY
# ============================================================

history_exists = os.path.exists(
    HISTORY_FILE
)

history_file = open(
    HISTORY_FILE,
    "a",
    newline="",
    encoding="utf-8",
)

history_writer = csv.writer(
    history_file
)

if not history_exists:
    history_writer.writerow(
        [
            "step",
            "train_loss",
            "val_loss",
            "lr",
            "grad_norm",
            "max_weight",
            "elapsed_sec",
        ]
    )


# ============================================================
# TRAIN EXTENSION
# ============================================================

print("=" * 64)
print(
    f"Continuing Glyph v2: "
    f"{start_step:,} -> {TARGET_STEP:,}"
)
print("=" * 64)
print()

running_loss = 0.0
start_time = time.time()
last_completed_step = start_step

for step in range(
    start_step + 1,
    TARGET_STEP + 1,
):

    model.train()

    # Keep the LR fixed at the final LR of the original
    # 20k-step schedule.
    for group in opt.param_groups:
        group["lr"] = CONTINUATION_LR

    x, y = get_batch(
        train_tensor,
        BATCH,
    )

    opt.zero_grad(
        set_to_none=True
    )

    _, loss = model(
        x,
        y,
    )

    if not torch.isfinite(loss):
        print()
        print(
            f"STOP: non-finite loss "
            f"at step {step}: {loss}"
        )
        break

    loss.backward()

    if not gradients_are_finite():
        print()
        print(
            f"STOP: non-finite gradient "
            f"at step {step}"
        )
        break

    grad_norm = (
        torch.nn.utils
        .clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )
    )

    if not torch.isfinite(
        grad_norm
    ):
        print()
        print(
            f"STOP: non-finite gradient norm "
            f"at step {step}"
        )
        break

    opt.step()

    if not parameters_are_finite():
        print()
        print(
            f"STOP: non-finite model parameter "
            f"at step {step}"
        )
        break

    running_loss += loss.item()
    last_completed_step = step

    # --------------------------------------------------------
    # Logging / validation
    # --------------------------------------------------------

    if step % EVAL_EVERY == 0:

        avg_train_loss = (
            running_loss
            / EVAL_EVERY
        )

        running_loss = 0.0

        val_loss = evaluate_validation()

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"step {step:5d} | "
            f"train {avg_train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"lr {CONTINUATION_LR:.2e} | "
            f"grad {float(grad_norm):.3f} | "
            f"maxW {max_weight():.2f} | "
            f"time {elapsed / 60:.1f}m"
        )

        history_writer.writerow(
            [
                step,
                f"{avg_train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{CONTINUATION_LR:.8e}",
                f"{float(grad_norm):.6f}",
                f"{max_weight():.6f}",
                f"{elapsed:.2f}",
            ]
        )

        history_file.flush()

        is_best = (
            val_loss < best_val_loss
        )

        if is_best:
            best_val_loss = val_loss

            print(
                "  -> new best "
                f"validation loss: "
                f"{best_val_loss:.6f}"
            )

        if (
            step % SAVE_EVERY == 0
            or is_best
        ):
            save_checkpoint(
                step,
                best_val_loss,
                best=is_best,
            )

# ============================================================
# FINAL SAVE
# ============================================================

if parameters_are_finite():

    save_checkpoint(
        last_completed_step,
        best_val_loss,
        best=False,
    )

history_file.close()

print()
print("=" * 64)
print("Glyph v2 extended training finished")
print("=" * 64)
print(
    f"Final step      : "
    f"{last_completed_step:,}"
)
print(
    f"Best val loss   : "
    f"{best_val_loss:.6f}"
)
print(
    f"Final checkpoint: {CKPT}"
)
print(
    f"Best checkpoint : {BEST_CKPT}"
)
print(
    f"History         : {HISTORY_FILE}"
)
print("=" * 64)
