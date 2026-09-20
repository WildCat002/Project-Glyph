import os
import time
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CPU SETTINGS
# ============================================================

torch.set_num_threads(4)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# DATA
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT_DIR / "data/data.txt"
CKPT = ROOT_DIR / "models/v1/glyph.pt"
BACKUP_CKPT = ROOT_DIR / "models/v1/glyph_last_good.pt"

MAX_DATA = 5_000_000

text = open(
    DATA_FILE,
    "r",
    encoding="utf-8",
    errors="ignore"
).read().lower()[:MAX_DATA]

chars = sorted(set(text))

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

V = len(chars)

data = np.array(
    [stoi[c] for c in text],
    dtype=np.uint16
)

print(f"Vocab {V}, tokens {len(data):,}")


# ============================================================
# CONFIG
# ============================================================

BLOCK = 64

N_LAYER = 3
N_HEAD = 4
N_EMB = 128

DROPOUT = 0.1

BATCH = 32

MAX_STEPS = 20_000

# Longer warmup
WARMUP = 4000

# Lower peak learning rate
LR = 1e-5

# Less aggressive weight decay
WD = 0.01

# Safer gradient clipping
GRAD_CLIP = 0.05

# More conservative Adam epsilon
ADAM_EPS = 1e-6

SAVE_EVERY = 500
EVAL_EVERY = 100

# IMPORTANT:
# Set this to False for the first run with this fixed code.
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
            bias=False
        )

        self.proj = nn.Linear(
            N_EMB,
            N_EMB,
            bias=False
        )

        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(x).split(N_EMB, dim=2)

        head_dim = C // N_HEAD

        q = q.view(
            B,
            T,
            N_HEAD,
            head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            N_HEAD,
            head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            N_HEAD,
            head_dim
        ).transpose(1, 2)

        # Explicit float32 attention.
        #
        # This avoids accidentally doing attention in a lower
        # precision type if the implementation/environment changes.
        q = q.float()
        k = k.float()
        v = v.float()

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        y = self.proj(y)

        return self.drop(y)


class Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(N_EMB)

        self.attn = Attn()

        self.ln2 = nn.LayerNorm(N_EMB)

        self.mlp = nn.Sequential(
            nn.Linear(
                N_EMB,
                4 * N_EMB
            ),

            nn.GELU(),

            nn.Linear(
                4 * N_EMB,
                N_EMB
            ),

            nn.Dropout(DROPOUT),
        )

        # Residual projection scaling.
        #
        # These layers are initialized separately AFTER the normal
        # initialization, so self.apply() does not overwrite them.
        scale = 1.0 / math.sqrt(2 * N_LAYER)

        nn.init.normal_(
            self.attn.proj.weight,
            mean=0.0,
            std=0.02 * scale
        )

        nn.init.normal_(
            self.mlp[2].weight,
            mean=0.0,
            std=0.02 * scale
        )

        if self.attn.proj.bias is not None:
            nn.init.zeros_(self.attn.proj.bias)

        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

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
            N_EMB
        )

        self.pos = nn.Embedding(
            BLOCK,
            N_EMB
        )

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(N_LAYER)
            ]
        )

        self.ln_f = nn.LayerNorm(N_EMB)

        self.head = nn.Linear(
            N_EMB,
            V,
            bias=False
        )

        # Normal initialization FIRST.
        self._initialize_weights()

        # Weight tying AFTER initialization.
        #
        # This makes both layers point to the same Parameter.
        self.head.weight = self.tok.weight

        # Special residual initialization LAST.
        scale = 1.0 / math.sqrt(2 * N_LAYER)

        for block in self.blocks:

            nn.init.normal_(
                block.attn.proj.weight,
                mean=0.0,
                std=0.02 * scale
            )

            nn.init.normal_(
                block.mlp[2].weight,
                mean=0.0,
                std=0.02 * scale
            )

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > BLOCK:
            raise ValueError(
                f"Sequence length {T} exceeds BLOCK={BLOCK}"
            )

        pos = torch.arange(
            T,
            device=idx.device
        )

        x = (
            self.tok(idx)
            +
            self.pos(pos)
        )

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.head(x)

        if targets is None:
            return logits, None

        # Cross entropy in float32.
        loss = F.cross_entropy(
            logits.float().reshape(-1, V),
            targets.reshape(-1)
        )

        return logits, loss


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

model = Glyph().to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Params: {num_params / 1e6:.2f}M"
)


# ============================================================
# PARAMETER GROUPS
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


print(
    "Decay params: "
    f"{sum(p.numel() for p in decay)}, "
    "No-decay: "
    f"{sum(p.numel() for p in no_decay)}"
)


# ============================================================
# OPTIMIZER
# ============================================================

opt = torch.optim.AdamW(
    [
        {
            "params": decay,
            "weight_decay": WD
        },
        {
            "params": no_decay,
            "weight_decay": 0.0
        },
    ],

    lr=LR,

    betas=(0.9, 0.95),

    eps=ADAM_EPS
)


# ============================================================
# LEARNING RATE
# ============================================================

def lr_at(step):

    # Linear warmup
    if step <= WARMUP:

        return LR * step / WARMUP

    # Progress after warmup
    progress = (
        step - WARMUP
    ) / (
        MAX_STEPS - WARMUP
    )

    progress = min(
        1.0,
        max(0.0, progress)
    )

    # Cosine decay from LR -> 10% LR
    return LR * (
        0.1
        +
        0.9
        *
        0.5
        *
        (
            1.0
            +
            math.cos(
                math.pi * progress
            )
        )
    )


# ============================================================
# FINITE CHECK
# ============================================================

def parameters_are_finite():

    for param in model.parameters():

        if not torch.isfinite(param).all():
            return False

    return True


def gradients_are_finite():

    for param in model.parameters():

        if param.grad is None:
            continue

        if not torch.isfinite(param.grad).all():
            return False

    return True


# ============================================================
# BATCH
# ============================================================

def get_batch(bs):

    ix = np.random.randint(
        0,
        len(data) - BLOCK - 1,
        size=bs
    )

    x = np.stack(
        [
            data[i:i + BLOCK]
            for i in ix
        ]
    ).astype(np.int64)

    y = np.stack(
        [
            data[i + 1:i + BLOCK + 1]
            for i in ix
        ]
    ).astype(np.int64)

    return (
        torch.from_numpy(x),
        torch.from_numpy(y)
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def save_checkpoint(step):

    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "vocab": chars,
        "config": {
            "BLOCK": BLOCK,
            "N_LAYER": N_LAYER,
            "N_HEAD": N_HEAD,
            "N_EMB": N_EMB,
            "BATCH": BATCH,
            "LR": LR,
            "WD": WD,
        }
    }

    # Save main checkpoint
    temp_file = CKPT + ".tmp"

    torch.save(
        state,
        temp_file
    )

    # Atomic replacement
    os.replace(
        temp_file,
        CKPT
    )

    # Keep a separate known-good backup
    try:
        shutil.copy2(
            CKPT,
            BACKUP_CKPT
        )
    except OSError:
        pass


# ============================================================
# RESUME
# ============================================================

start_step = 0

if RESUME and os.path.exists(CKPT):

    print(
        f"Loading checkpoint: {CKPT}"
    )

    ck = torch.load(
        CKPT,
        map_location=device
    )

    try:

        model.load_state_dict(
            ck["model"]
        )

        opt.load_state_dict(
            ck["opt"]
        )

        start_step = int(
            ck["step"]
        )

        print(
            f"Resumed from step {start_step}"
        )

        # Never continue from a corrupted model.
        if not parameters_are_finite():

            print(
                "Checkpoint contains non-finite "
                "model parameters."
            )

            print(
                "Starting from fresh weights."
            )

            model = Glyph().to(device)

            opt = torch.optim.AdamW(
                [
                    {
                        "params": [
                            p
                            for p in model.parameters()
                            if p.requires_grad
                            and p.dim() >= 2
                        ],
                        "weight_decay": WD
                    },
                    {
                        "params": [
                            p
                            for p in model.parameters()
                            if p.requires_grad
                            and p.dim() < 2
                        ],
                        "weight_decay": 0.0
                    },
                ],
                lr=LR,
                betas=(0.9, 0.95),
                eps=ADAM_EPS
            )

            start_step = 0

    except Exception as e:

        print(
            f"Could not resume checkpoint: {e}"
        )

        print(
            "Starting from fresh weights."
        )

        model = Glyph().to(device)

        opt = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for p in model.parameters()
                        if p.requires_grad
                        and p.dim() >= 2
                    ],
                    "weight_decay": WD
                },
                {
                    "params": [
                        p
                        for p in model.parameters()
                        if p.requires_grad
                        and p.dim() < 2
                    ],
                    "weight_decay": 0.0
                },
            ],
            lr=LR,
            betas=(0.9, 0.95),
            eps=ADAM_EPS
        )

        start_step = 0


# ============================================================
# TRAINING
# ============================================================

if start_step >= MAX_STEPS:

    print(
        f"Training already complete at step "
        f"{start_step}."
    )

    raise SystemExit(0)


model.train()

running_loss = 0.0

t0 = time.time()

last_good_step = start_step


print()
print("Training Glyph...")
print(
    f"Peak LR: {LR:.2e}"
)
print(
    f"Warmup: {WARMUP} steps"
)
print(
    f"Gradient clip: {GRAD_CLIP}"
)
print()


for step in range(
    start_step + 1,
    MAX_STEPS + 1
):

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    current_lr = lr_at(step)

    for group in opt.param_groups:
        group["lr"] = current_lr


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    xb, yb = get_batch(BATCH)

    xb = xb.to(device)
    yb = yb.to(device)


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    _, loss = model(
        xb,
        yb
    )


    # --------------------------------------------------------
    # Loss safety check
    # --------------------------------------------------------

    if not torch.isfinite(loss):

        print()
        print(
            f"NON-FINITE LOSS at step {step}"
        )

        print(
            f"Loss: {loss.item()}"
        )

        print(
            f"Last good step: {last_good_step}"
        )

        print(
            "Training stopped before optimizer update."
        )

        break


    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    opt.zero_grad(
        set_to_none=True
    )

    loss.backward()


    # --------------------------------------------------------
    # Gradient safety check
    # --------------------------------------------------------

    if not gradients_are_finite():

        print()
        print(
            f"NON-FINITE GRADIENT at step {step}"
        )

        print(
            f"Last good step: {last_good_step}"
        )

        print(
            "Training stopped before optimizer update."
        )

        break


    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
        error_if_nonfinite=True
    )


    # Extra check after clipping
    if not torch.isfinite(grad_norm):

        print()
        print(
            f"NON-FINITE GRADIENT NORM at step {step}"
        )

        print(
            f"Last good step: {last_good_step}"
        )

        break


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    opt.step()


    # --------------------------------------------------------
    # Model safety check
    # --------------------------------------------------------

    if not parameters_are_finite():

        print()
        print(
            f"NON-FINITE MODEL PARAMETERS at step {step}"
        )

        print(
            f"Last good step: {last_good_step}"
        )

        print(
            "Optimizer update produced invalid weights."
        )

        break


    # --------------------------------------------------------
    # Successful step
    # --------------------------------------------------------

    last_good_step = step

    running_loss += loss.item()


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if step % EVAL_EVERY == 0:

        maxW = max(
            p.detach()
             .abs()
             .max()
             .item()
            for p in model.parameters()
        )

        maxG = max(
            (
                p.grad.detach()
                 .abs()
                 .max()
                 .item()
                if p.grad is not None
                else 0.0
            )
            for p in model.parameters()
        )

        elapsed = (
            time.time() - t0
        )

        print(
            f"step {step:5d}  "
            f"loss {running_loss / EVAL_EVERY:.4f}  "
            f"lr {current_lr:.2e}  "
            f"maxW {maxW:.2f}  "
            f"maxG {maxG:.3f}  "
            f"gradNorm {grad_norm.item():.3f}  "
            f"{elapsed:.1f}s"
        )

        running_loss = 0.0
        t0 = time.time()


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if step % SAVE_EVERY == 0:

        save_checkpoint(step)

        print(
            "  saved"
        )


# ============================================================
# FINAL SAVE
# ============================================================

if last_good_step > start_step:

    save_checkpoint(last_good_step)

    print()
    print(
        f"Stopped/saved safely at step "
        f"{last_good_step}."
    )

else:

    print()
    print(
        "No new successful training steps."
    )