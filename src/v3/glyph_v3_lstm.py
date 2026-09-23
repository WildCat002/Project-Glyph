"""
Glyph v3-A: same-budget recurrent language model baseline.

Purpose
-------
V3-A is a controlled architecture experiment against Glyph v2.

The dataset split, vocabulary, sequence length, and approximate parameter
budget are kept the same as V2. The main architectural change is:

    V2: decoder-only Transformer
    V3-A: 2-layer LSTM

The model is deliberately small and inspectable. This script is the training
entry point; the interactive generator is in glyph_v3_chat.py.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================

SEED = 42
CPU_THREADS = 4

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

torch.set_num_threads(CPU_THREADS)
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATHS / DATA
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data/data.txt"

MAX_DATA = 5_000_000
VAL_CHARS = 250_000
TRAIN_END = MAX_DATA - VAL_CHARS

# The V2 split is intentionally preserved for the first V3-A comparison.
raw_text = DATA_FILE.read_text(encoding="utf-8", errors="ignore").lower()
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
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
V = len(chars)

# int64 avoids the CUDA UInt16 advanced-indexing problem seen in V2.
train_data = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
val_data = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)

# ============================================================
# CONFIG
# ============================================================

SEQ_LEN = 128
EMB = 96
HIDDEN = 240
N_LAYERS = 2
DROPOUT = 0.10
BATCH = 32

# ~820K params, intentionally almost identical to V2's 820,224.
MAX_STEPS = 100_000
WARMUP = 2_000
LR = 3e-4
MIN_LR = 3e-5
WD = 0.01
GRAD_CLIP = 1.0
ADAM_EPS = 1e-8

SAVE_EVERY = 1_000
EVAL_EVERY = 1_000
VAL_BATCHES = 128

MODEL_PATH = ROOT_DIR / "models/v3/glyph_v3_lstm.pt"
BEST_PATH = ROOT_DIR / "models/v3/glyph_v3_lstm_best.pt"
LAST_GOOD_PATH = ROOT_DIR / "experiments/v3_lstm/glyph_v3_lstm_last_good.pt"
HISTORY_PATH = ROOT_DIR / "experiments/v3_lstm/glyph_v3_lstm_history.csv"

RESUME = False


# ============================================================
# MODEL
# ============================================================


class GlyphV3LSTM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.tok = nn.Embedding(vocab_size, EMB)
        self.rnn = nn.LSTM(
            input_size=EMB,
            hidden_size=HIDDEN,
            num_layers=N_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.head = nn.Linear(HIDDEN, vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)

        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget gate bias = +1.
                hidden = HIDDEN
                param.data[hidden : 2 * hidden].fill_(1.0)

        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor, hidden=None):
        x = self.tok(idx)
        x, hidden = self.rnn(x, hidden)
        logits = self.head(x)
        return logits, hidden


# ============================================================
# DATA / EVALUATION
# ============================================================


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    # Keep the dataset on CPU and use vectorized contiguous window indexing.
    # This avoids the UInt16 CUDA indexing issue that appeared in the V2 run
    # while avoiding a Python loop for every batch.
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    offsets = torch.arange(seq_len + 1)
    windows = data[starts[:, None] + offsets[None, :]]
    x = windows[:, :-1].to(device)
    y = windows[:, 1:].to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, batches: int, device: torch.device):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, BATCH, SEQ_LEN, device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, V).float(),
            y.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def cosine_lr(step: int):
    if step <= 0:
        return LR / max(WARMUP, 1)
    if step < WARMUP:
        return LR * step / WARMUP

    progress = (step - WARMUP) / max(MAX_STEPS - WARMUP, 1)
    progress = min(max(progress, 0.0), 1.0)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model, optimizer, step, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {
            "seq_len": SEQ_LEN,
            "emb": EMB,
            "hidden": HIDDEN,
            "layers": N_LAYERS,
            "dropout": DROPOUT,
            "batch": BATCH,
            "max_steps": MAX_STEPS,
            "warmup": WARMUP,
            "lr": LR,
            "min_lr": MIN_LR,
            "weight_decay": WD,
            "grad_clip": GRAD_CLIP,
            "adam_eps": ADAM_EPS,
            "train_chars": len(train_data),
            "val_chars": len(val_data),
            "vocab": V,
            "chars": chars,
        },
    }
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])
    return int(checkpoint.get("step", 0)), float(checkpoint.get("best_val_loss", float("inf")))


def count_parameters(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters())


# ============================================================
# TRAIN
# ============================================================


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_GOOD_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = GlyphV3LSTM(V).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
        eps=ADAM_EPS,
    )

    start_step = 0
    best_val_loss = float("inf")

    if RESUME and MODEL_PATH.exists():
        start_step, best_val_loss = load_checkpoint(MODEL_PATH, model, optimizer)
        print(f"Resumed from step {start_step}")

    params = count_parameters(model)

    print("=" * 68)
    print("Glyph v3-A LSTM Training")
    print("=" * 68)
    print(f"Device       : {DEVICE}")
    print(f"Total chars  : {len(text):,}")
    print(f"Train chars  : {len(train_data):,}")
    print(f"Val chars    : {len(val_data):,}")
    print(f"Vocabulary   : {V}")
    print(f"Sequence     : {SEQ_LEN}")
    print(f"Embedding    : {EMB}")
    print(f"Hidden       : {HIDDEN}")
    print(f"Layers       : {N_LAYERS}")
    print(f"Parameters   : {params:,}")
    print(f"Max steps    : {MAX_STEPS:,}")
    print(f"Learning rate: {LR:g} -> {MIN_LR:g}")
    print("=" * 68)

    if not HISTORY_PATH.exists() or start_step == 0:
        with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "max_weight"])

    last_log = time.time()

    for step in range(start_step + 1, MAX_STEPS + 1):
        lr_now = cosine_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        model.train()
        x, y = get_batch(train_data, BATCH, SEQ_LEN, DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm}")

        optimizer.step()

        max_weight = max(
            parameter.detach().abs().max().item()
            for parameter in model.parameters()
            if parameter.numel()
        )

        if step == 1 or step % EVAL_EVERY == 0:
            train_loss = estimate_loss(model, train_data, 32, DEVICE)
            val_loss = estimate_loss(model, val_data, VAL_BATCHES, DEVICE)

            with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([
                    step,
                    f"{train_loss:.8f}",
                    f"{val_loss:.8f}",
                    f"{lr_now:.10g}",
                    f"{float(grad_norm):.8f}",
                    f"{max_weight:.8f}",
                ])

            elapsed = time.time() - last_log
            last_log = time.time()

            print(
                f"step {step:6d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| lr {lr_now:.3e} | grad {float(grad_norm):.3f} "
                f"| maxW {max_weight:.3f} | {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(BEST_PATH, model, optimizer, step, best_val_loss)
                print(f"  saved BEST -> {BEST_PATH}")

        if step == 1 or step % SAVE_EVERY == 0:
            save_checkpoint(MODEL_PATH, model, optimizer, step, best_val_loss)
            # This file is a recovery checkpoint, not a named release model.
            save_checkpoint(LAST_GOOD_PATH, model, optimizer, step, best_val_loss)

    print("=" * 68)
    print("Training complete")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint     : {BEST_PATH}")
    print(f"Final checkpoint    : {MODEL_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
