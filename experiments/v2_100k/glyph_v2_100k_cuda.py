"""
Glyph v2 Extended Training — CUDA / Google Colab T4
====================================================

Continue the EXISTING Glyph v2 50,000-step checkpoint to step 100,000
using a CUDA GPU (for example, the Google Colab NVIDIA T4).

This keeps the same model architecture and training schedule as the
100k continuation, but moves the model and the full train/validation
character tensors onto the GPU so the T4 actually performs the work.

Architecture:
    BLOCK       = 128
    N_LAYER     = 4
    N_HEAD      = 4
    N_EMB       = 128
    BATCH       = 32
    WD          = 0.01
    GRAD_CLIP   = 0.05
    ADAM_EPS    = 1e-6

Continuation schedule:
    50,000 -> 52,000 : LR ramps 1e-6 -> 3e-6
    52,000 -> 100,000: LR held at 3e-6

Source checkpoint:
    glyph_v2_50k.pt

CUDA outputs are kept separate so they do not conflict with a local
laptop 100k run:
    glyph_v2_100k_cuda.pt
    glyph_v2_100k_cuda_best.pt
    glyph_v2_100k_cuda_last_good.pt
    glyph_v2_100k_cuda_history.csv

Notes:
    - The trainer uses CUDA when available and stops with an error if CUDA
      is unavailable. This prevents silently falling back to CPU.
    - Training stays in float32 for closer numerical behavior to the
      existing CPU experiment. The T4 still performs the matrix operations.
    - The first step can be slower because CUDA kernels/context are warming up.
    - Checkpoint tensors are saved normally; they can later be loaded on CPU
      with map_location='cpu'.
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# FILES
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data/data.txt"
BASE_CKPT = ROOT_DIR / "experiments/v2_50k/glyph_v2_50k.pt"

CKPT = ROOT_DIR / "experiments/v2_100k/glyph_v2_100k_cuda.pt"
BEST_CKPT = ROOT_DIR / "models/v2/glyph_v2_100k_cuda_best.pt"
LAST_GOOD_CKPT = ROOT_DIR / "experiments/v2_100k/glyph_v2_100k_cuda_last_good.pt"
HISTORY_FILE = ROOT_DIR / "experiments/v2_100k/glyph_v2_100k_cuda_history.csv"


# ============================================================
# CUDA DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. In Google Colab, enable a GPU runtime "
        "(Runtime -> Change runtime type -> T4 GPU) and run again."
    )

DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_PROPS = torch.cuda.get_device_properties(0)

print("=" * 64)
print("Glyph v2 Extended Training — CUDA")
print("=" * 64)
print(f"Device       : {DEVICE}")
print(f"GPU          : {GPU_NAME}")
print(f"VRAM         : {GPU_PROPS.total_memory / (1024**3):.2f} GB")
print(f"PyTorch      : {torch.__version__}")
print("=" * 64)
print()


# ============================================================
# DATA
# ============================================================

MAX_DATA = 5_000_000
VAL_CHARS = 250_000
TRAIN_END = MAX_DATA - VAL_CHARS

with open(
    DATA_FILE,
    "r",
    encoding="utf-8",
    errors="ignore",
) as f:
    raw_text = f.read().lower()

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

print(f"Total chars  : {len(text):,}")
print(f"Train chars  : {len(train_data):,}")
print(f"Val chars    : {len(val_data):,}")
print(f"Vocab        : {V}")
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
SOURCE_STEP_REQUIRED = 50_000
TARGET_STEP = 100_000

CONTINUATION_START_LR = 1e-6
CONTINUATION_LR = 3e-6
LR_RAMP_STEPS = 2_000

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

        # Keep the attention calculation in float32 to match the existing
        # CPU trainer numerically as closely as practical.
        q = (
            q.view(B, T, N_HEAD, head_dim)
            .transpose(1, 2)
            .float()
        )

        k = (
            k.view(B, T, N_HEAD, head_dim)
            .transpose(1, 2)
            .float()
        )

        v = (
            v.view(B, T, N_HEAD, head_dim)
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
    def __init__(self):
        super().__init__()

        self.tok = nn.Embedding(V, N_EMB)
        self.pos = nn.Embedding(BLOCK, N_EMB)

        self.blocks = nn.Sequential(
            *[Block() for _ in range(N_LAYER)]
        )

        self.ln_f = nn.LayerNorm(N_EMB)
        self.head = nn.Linear(N_EMB, V, bias=False)

        self._initialize_weights()

        # Weight tying, same as original v2.
        self.head.weight = self.tok.weight

        # Same residual projection scaling as original v2.
        scale = 1.0 / math.sqrt(2 * N_LAYER)

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
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > BLOCK:
            raise ValueError(
                f"Sequence length {T} exceeds BLOCK={BLOCK}"
            )

        pos = torch.arange(
            T,
            device=idx.device,
        )

        x = self.tok(idx) + self.pos(pos)
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

model = Glyph().to(DEVICE)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters   : {num_params:,} ({num_params / 1e6:.2f}M)")
print(f"Context      : {BLOCK}")
print(f"Layers       : {N_LAYER}")
print(f"Heads        : {N_HEAD}")
print(f"Embedding    : {N_EMB}")


decay = []
no_decay = []

for _, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if param.dim() >= 2:
        decay.append(param)
    else:
        no_decay.append(param)

print(f"Decay params    : {sum(p.numel() for p in decay):,}")
print(f"No-decay params : {sum(p.numel() for p in no_decay):,}")

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
# MOVE DATA TO GPU
# ============================================================

# The entire corpus fits comfortably in T4 VRAM.
# CUDA does not support advanced indexing on UInt16 tensors, so store
# the token IDs directly as int64 on the GPU. This is still tiny:
# about 40 MB for train and 2 MB for validation.
train_tensor = torch.from_numpy(train_data).to(
    DEVICE,
    dtype=torch.long,
    non_blocking=True,
)

val_tensor = torch.from_numpy(val_data).to(
    DEVICE,
    dtype=torch.long,
    non_blocking=True,
)

print(
    f"GPU data      : train {train_tensor.numel():,} chars, "
    f"val {val_tensor.numel():,} chars"
)
print()


# ============================================================
# GPU BATCHING
# ============================================================


def get_batch(source, bs):
    if source.numel() <= BLOCK + 1:
        raise ValueError(
            "Dataset split is too small for the context length."
        )

    # Generate random start positions directly on the GPU.
    ix = torch.randint(
        0,
        source.numel() - BLOCK - 1,
        (bs,),
        device=DEVICE,
    )

    # Build the [B, BLOCK + 1] windows directly on the GPU.
    offsets = torch.arange(
        BLOCK + 1,
        device=DEVICE,
    )

    positions = ix[:, None] + offsets[None, :]
    batch = source[positions]

    return batch[:, :-1], batch[:, 1:]


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

        if not torch.isfinite(param.grad).all():
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
                .item()
            )
        )

    return max(values) if values else 0.0


# ============================================================
# VALIDATION
# ============================================================


@torch.no_grad()
def evaluate_validation(num_batches=VAL_BATCHES):
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = get_batch(val_tensor, BATCH)
        _, loss = model(x, y)

        if not torch.isfinite(loss):
            model.train()
            return float("inf")

        total_loss += loss.item()

    model.train()
    return total_loss / num_batches


# ============================================================
# CHECKPOINTS
# ============================================================


def make_state(step, best_val_loss):
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {
            "run": "glyph_v2_extended_100k_cuda",
            "base_checkpoint": BASE_CKPT,
            "source_step": SOURCE_STEP_REQUIRED,
            "device": "cuda",
            "gpu_name": GPU_NAME,
            "eval_every": EVAL_EVERY,
            "val_batches": VAL_BATCHES,
            "save_every": SAVE_EVERY,
            "batching": "gpu_torch_indexing",
            "target_step": TARGET_STEP,
            "continuation_start_lr": CONTINUATION_START_LR,
            "continuation_lr": CONTINUATION_LR,
            "lr_ramp_steps": LR_RAMP_STEPS,
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
            "float32_training": True,
        },
    }


def atomic_torch_save(state, path):
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_checkpoint(step, best_val_loss, best=False):
    state = make_state(step, best_val_loss)

    atomic_torch_save(state, CKPT)
    atomic_torch_save(state, LAST_GOOD_CKPT)

    if best:
        atomic_torch_save(state, BEST_CKPT)


# ============================================================
# LEARNING-RATE SCHEDULE
# ============================================================


def lr_for_step(step):
    progress = step - SOURCE_STEP_REQUIRED

    if progress <= 0:
        return CONTINUATION_START_LR

    if progress >= LR_RAMP_STEPS:
        return CONTINUATION_LR

    alpha = progress / LR_RAMP_STEPS

    return CONTINUATION_START_LR + alpha * (
        CONTINUATION_LR - CONTINUATION_START_LR
    )


# ============================================================
# LOAD / RESUME CHECKPOINT
# ============================================================

resume_path = None
completed_target = False

if os.path.exists(CKPT):
    try:
        probe = torch.load(
            CKPT,
            map_location="cpu",
            weights_only=False,
        )

        probe_step = int(probe.get("step", 0))

        if probe_step >= TARGET_STEP:
            completed_target = True
        elif SOURCE_STEP_REQUIRED <= probe_step < TARGET_STEP:
            resume_path = CKPT

    except Exception:
        resume_path = None

if completed_target:
    print("Glyph v2 CUDA is already trained through 100,000 steps.")
    print(f"Checkpoint: {CKPT}")
    raise SystemExit(0)

checkpoint_path = (
    resume_path
    if resume_path is not None
    else BASE_CKPT
)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Missing checkpoint: {checkpoint_path}. "
        "Upload glyph_v2_50k.pt to the Colab working directory."
    )

print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

source_step = int(checkpoint.get("step", 0))

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

source_config = checkpoint.get("config", {})

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
    actual_value = source_config.get(key)

    if actual_value != expected_value:
        raise RuntimeError(
            f"Checkpoint config mismatch for {key}: "
            f"expected {expected_value!r}, found {actual_value!r}."
        )

model.load_state_dict(checkpoint["model"])

opt.load_state_dict(checkpoint["opt"])

# Optimizer states loaded from a CPU checkpoint need to be moved to CUDA.
for state in opt.state.values():
    for key, value in list(state.items()):
        if torch.is_tensor(value):
            state[key] = value.to(DEVICE)

for group in opt.param_groups:
    group["lr"] = lr_for_step(source_step)

start_step = source_step
best_val_loss = float(
    checkpoint.get("best_val_loss", float("inf"))
)

print(f"Loaded step     : {start_step:,}")
print(f"Best val loss   : {best_val_loss:.6f}")
print(
    f"LR schedule     : {CONTINUATION_START_LR:.2e} -> "
    f"{CONTINUATION_LR:.2e} over {LR_RAMP_STEPS:,} steps"
)
print(f"GPU             : {GPU_NAME}")
print()


# ============================================================
# SAFETY / CUDA WARM-UP
# ============================================================

if not parameters_are_finite():
    raise RuntimeError(
        "Source checkpoint contains non-finite model parameters."
    )

print("Running CUDA warm-up...")

with torch.no_grad():
    warm_x, _ = get_batch(train_tensor, BATCH)
    _ = model(warm_x)

torch.cuda.synchronize()

anchor_val = evaluate_validation()
torch.cuda.synchronize()

print(
    f"Validation at step {start_step:,}: "
    f"{anchor_val:.6f}"
)

if not math.isfinite(anchor_val):
    raise RuntimeError(
        "Source checkpoint produced non-finite validation loss."
    )

print()


# ============================================================
# HISTORY
# ============================================================

history_exists = os.path.exists(HISTORY_FILE)

history_file = open(
    HISTORY_FILE,
    "a",
    newline="",
    encoding="utf-8",
)

history_writer = csv.writer(history_file)

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
# TRAIN
# ============================================================

print("=" * 64)
print(
    f"Continuing Glyph v2 on {GPU_NAME}: "
    f"{start_step:,} -> {TARGET_STEP:,}"
)
print("=" * 64)
print()

running_loss = 0.0
start_time = time.time()
last_completed_step = start_step

for step in range(start_step + 1, TARGET_STEP + 1):
    model.train()

    current_lr = lr_for_step(step)

    for group in opt.param_groups:
        group["lr"] = current_lr

    x, y = get_batch(train_tensor, BATCH)

    opt.zero_grad(set_to_none=True)

    _, loss = model(x, y)

    if not torch.isfinite(loss):
        print()
        print(f"STOP: non-finite loss at step {step}: {loss}")
        break

    loss.backward()

    if not gradients_are_finite():
        print()
        print(f"STOP: non-finite gradient at step {step}")
        break

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    if not torch.isfinite(grad_norm):
        print()
        print(f"STOP: non-finite gradient norm at step {step}")
        break

    opt.step()

    if not parameters_are_finite():
        print()
        print(f"STOP: non-finite model parameter at step {step}")
        break

    running_loss += loss.item()
    last_completed_step = step

    if step % EVAL_EVERY == 0:
        # Make sure all GPU work is complete before measuring elapsed time.
        torch.cuda.synchronize()

        avg_train_loss = running_loss / EVAL_EVERY
        running_loss = 0.0

        val_loss = evaluate_validation()
        torch.cuda.synchronize()

        elapsed = time.time() - start_time

        print(
            f"step {step:5d} | "
            f"train {avg_train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"lr {current_lr:.2e} | "
            f"grad {float(grad_norm):.3f} | "
            f"maxW {max_weight():.2f} | "
            f"time {elapsed / 60:.1f}m"
        )

        history_writer.writerow(
            [
                step,
                f"{avg_train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{current_lr:.8e}",
                f"{float(grad_norm):.6f}",
                f"{max_weight():.6f}",
                f"{elapsed:.2f}",
            ]
        )
        history_file.flush()

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            print(
                "  -> new best validation loss: "
                f"{best_val_loss:.6f}"
            )

        if step % SAVE_EVERY == 0 or is_best:
            save_checkpoint(
                step,
                best_val_loss,
                best=is_best,
            )


# ============================================================
# FINAL SAVE
# ============================================================

torch.cuda.synchronize()

if parameters_are_finite():
    save_checkpoint(
        last_completed_step,
        best_val_loss,
        best=False,
    )

history_file.close()

torch.cuda.empty_cache()

print()
print("=" * 64)
print("Glyph v2 100k CUDA continuation finished")
print("=" * 64)
print(f"Final step      : {last_completed_step:,}")
print(f"Best val loss   : {best_val_loss:.6f}")
print(f"Final checkpoint: {CKPT}")
print(f"Best checkpoint : {BEST_CKPT}")
print(f"History         : {HISTORY_FILE}")
print("=" * 64)
