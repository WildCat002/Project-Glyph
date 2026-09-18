import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ============================================================
# Glyph v2
# ============================================================
# Fresh v2 training run.
#
# Main changes from v1:
# - Proper train/validation split
# - Context length: 128 (v1: 64)
# - Layers: 4 (v1: 3)
# - Same embedding width/head count for a controlled capacity increase
# - Separate v2 checkpoints; v1 glyph.pt is never modified
# - Periodic validation + best-checkpoint tracking
# - Training history saved to CSV
# ============================================================


# ============================================================
# CPU / REPRODUCIBILITY
# ============================================================

SEED = 42
CPU_THREADS = 4

torch.set_num_threads(CPU_THREADS)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DATA
# ============================================================

DATA_FILE = "data.txt"

MAX_DATA = 5_000_000
VAL_CHARS = 250_000

# Use the first 4.75M chars for training and the final 250K for validation.
TRAIN_END = MAX_DATA - VAL_CHARS

raw_text = (
    open(
        DATA_FILE,
        "r",
        encoding="utf-8",
        errors="ignore",
    )
    .read()
    .lower()
)

text = raw_text[:MAX_DATA]

if len(text) < MAX_DATA:
    print(
        f"Warning: data.txt contains only {len(text):,} chars; "
        f"expected {MAX_DATA:,}."
    )

if len(text) <= VAL_CHARS + 1:
    raise ValueError("Not enough text for the requested train/validation split.")

train_text = text[:TRAIN_END]
val_text = text[TRAIN_END:]

chars = sorted(set(text))

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

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
print("Glyph v2 Training")
print("=" * 64)
print(f"Total chars : {len(text):,}")
print(f"Train chars : {len(train_data):,}")
print(f"Val chars   : {len(val_data):,}")
print(f"Vocab       : {V}")
print()


# ============================================================
# CONFIG
# ============================================================

BLOCK = 128

N_LAYER = 4
N_HEAD = 4
N_EMB = 128

DROPOUT = 0.1

BATCH = 32

MAX_STEPS = 20_000
WARMUP = 4_000

LR = 1e-5
WD = 0.01

GRAD_CLIP = 0.05
ADAM_EPS = 1e-6

SAVE_EVERY = 500
EVAL_EVERY = 500

# Approximate validation during training.
# Final benchmark can run a full validation pass.
VAL_BATCHES = 128

CKPT = "glyph_v2.pt"
BEST_CKPT = "glyph_v2_best.pt"
LAST_GOOD_CKPT = "glyph_v2_last_good.pt"
HISTORY_FILE = "glyph_v2_history.csv"

# Start v2 from scratch.
RESUME = False


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
    def __init__(self):
        super().__init__()

        self.tok = nn.Embedding(V, N_EMB)
        self.pos = nn.Embedding(BLOCK, N_EMB)

        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])

        self.ln_f = nn.LayerNorm(N_EMB)

        # Tied output head, same as v1.
        self.head = nn.Linear(
            N_EMB,
            V,
            bias=False,
        )

        self._initialize_weights()

        # Weight tying after initialization.
        self.head.weight = self.tok.weight

        # GPT-style scaled residual projection initialization.
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
            raise ValueError(f"Sequence length {T} exceeds BLOCK={BLOCK}")

        pos = torch.arange(T, device=idx.device)

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
# MODEL / DEVICE
# ============================================================

device = torch.device("cpu")

model = Glyph().to(device)

num_params = sum(p.numel() for p in model.parameters())

print(f"Parameters  : {num_params:,} ({num_params / 1e6:.2f}M)")
print(f"Context     : {BLOCK}")
print(f"Layers      : {N_LAYER}")
print(f"Heads       : {N_HEAD}")
print(f"Embedding   : {N_EMB}")
print()


# ============================================================
# OPTIMIZER
# ============================================================

decay = []
no_decay = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if param.dim() >= 2:
        decay.append(param)
    else:
        no_decay.append(param)

print("Decay params    : " f"{sum(p.numel() for p in decay):,}")

print("No-decay params : " f"{sum(p.numel() for p in no_decay):,}")

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
    lr=LR,
    betas=(0.9, 0.95),
    eps=ADAM_EPS,
)


# ============================================================
# LEARNING RATE
# ============================================================


def lr_at(step):
    if step <= WARMUP:
        return LR * step / WARMUP

    progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
    progress = min(1.0, max(0.0, progress))

    return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ============================================================
# BATCHES
# ============================================================


def get_batch(source, bs):
    if len(source) <= BLOCK + 1:
        raise ValueError("Dataset split is too small for the context length.")

    ix = np.random.randint(
        0,
        len(source) - BLOCK - 1,
        size=bs,
    )

    x = np.stack([source[i : i + BLOCK] for i in ix]).astype(np.int64)

    y = np.stack([source[i + 1 : i + BLOCK + 1] for i in ix]).astype(np.int64)

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
    )


# ============================================================
# FINITE CHECKS
# ============================================================


def parameters_are_finite():
    return all(torch.isfinite(param).all().item() for param in model.parameters())


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
        values.append(float(param.detach().abs().max()))

    return max(values) if values else 0.0


# ============================================================
# VALIDATION
# ============================================================


@torch.no_grad()
def evaluate_validation(num_batches=VAL_BATCHES):
    model.eval()

    total_loss = 0.0

    for _ in range(num_batches):
        x, y = get_batch(val_data, BATCH)

        _, loss = model(
            x.to(device),
            y.to(device),
        )

        if not torch.isfinite(loss):
            model.train()
            return float("inf")

        total_loss += float(loss)

    model.train()

    return total_loss / num_batches


# ============================================================
# CHECKPOINTING
# ============================================================


def make_state(step, best_val_loss):
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {
            "block": BLOCK,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_emb": N_EMB,
            "batch": BATCH,
            "max_steps": MAX_STEPS,
            "warmup": WARMUP,
            "lr": LR,
            "weight_decay": WD,
            "grad_clip": GRAD_CLIP,
            "adam_eps": ADAM_EPS,
            "train_chars": len(train_data),
            "val_chars": len(val_data),
            "vocab": V,
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
# RESUME
# ============================================================

start_step = 0
best_val_loss = float("inf")

if RESUME and os.path.exists(CKPT):
    print(f"Resuming from {CKPT}...")

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["opt"])

    start_step = int(checkpoint.get("step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    print(f"Resume step   : {start_step}")
    print(f"Best val loss : {best_val_loss:.6f}")
    print()


# ============================================================
# TRAINING HISTORY
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
print("Starting Glyph v2 training")
print("=" * 64)
print()

running_loss = 0.0
start_time = time.time()

for step in range(start_step + 1, MAX_STEPS + 1):

    model.train()

    current_lr = lr_at(step)

    for group in opt.param_groups:
        group["lr"] = current_lr

    x, y = get_batch(train_data, BATCH)

    opt.zero_grad(set_to_none=True)

    _, loss = model(
        x.to(device),
        y.to(device),
    )

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

    # --------------------------------------------------------
    # Periodic logging / validation
    # --------------------------------------------------------

    if step % EVAL_EVERY == 0 or step == 1:

        avg_train_loss = running_loss / (EVAL_EVERY if step != 1 else 1)

        running_loss = 0.0

        val_loss = evaluate_validation()

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

        if step % SAVE_EVERY == 0 or is_best:
            save_checkpoint(
                step,
                best_val_loss,
                best=is_best,
            )

        if is_best:
            print(f"  -> new best validation loss: " f"{best_val_loss:.4f}")

# Final save
final_step = min(
    MAX_STEPS,
    step if "step" in locals() else start_step,
)

if parameters_are_finite():
    save_checkpoint(
        final_step,
        best_val_loss,
        best=False,
    )

history_file.close()

print()
print("=" * 64)
print("Glyph v2 training finished")
print("=" * 64)
print(f"Final step        : {final_step}")
print(f"Best val loss     : {best_val_loss:.4f}")
print(f"Checkpoint        : {CKPT}")
print(f"Best checkpoint   : {BEST_CKPT}")
print(f"History           : {HISTORY_FILE}")
print("=" * 64)
