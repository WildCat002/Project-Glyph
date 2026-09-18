"""
speed_benchmark_v2.py

Benchmark the EXISTING Glyph v2 training workload without changing the
real 20k -> 50k experiment.

It measures:
    1. Eager PyTorch CPU training at several thread counts.
    2. torch.compile() on the fastest eager thread count, when supported.

The benchmark:
    - loads glyph_v2.pt
    - loads the same 5M-char dataset and 4.75M/250k split
    - uses the same model architecture
    - uses the same batch size and context length
    - uses the same AdamW hyperparameters
    - starts from the same trained checkpoint
    - times complete training steps (batch extraction + forward + backward
      + gradient clipping + optimizer step)
    - never writes back to glyph_v2.pt

This is a SPEED benchmark only. No benchmark checkpoint is saved.

Run from:
    E:\\Python-Projects\\Project-Glyph\\Glyph

    python speed_benchmark_v2.py

Optional:
    python speed_benchmark_v2.py --steps 120 --warmup 40
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


DATA_FILE = "data.txt"
CHECKPOINT = "glyph_v2.pt"

MAX_DATA = 5_000_000
VAL_CHARS = 250_000

BLOCK = 128
N_LAYER = 4
N_HEAD = 4
N_EMB = 128
DROPOUT = 0.1

BATCH = 32

LR = 1e-6
WD = 0.01
GRAD_CLIP = 0.05
ADAM_EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8",
        help="Comma-separated eager CPU thread counts.",
    )
    return p.parse_args()


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

        q, k, v = self.qkv(x).split(
            N_EMB,
            dim=2,
        )

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
    def __init__(self, vocab_size):
        super().__init__()

        self.tok = nn.Embedding(
            vocab_size,
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
            vocab_size,
            bias=False,
        )

        # Do not reinitialize after loading.
        self.head.weight = self.tok.weight

    def forward(self, idx, targets):
        B, T = idx.shape

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

        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        return loss


# ============================================================
# DATA
# ============================================================

def load_data():
    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        text = f.read().lower()

    text = text[:MAX_DATA]

    train_end = len(text) - VAL_CHARS

    if train_end <= BLOCK + 1:
        raise RuntimeError(
            "Not enough data for the configured split."
        )

    chars = sorted(set(text))

    stoi = {
        ch: i
        for i, ch in enumerate(chars)
    }

    encoded = np.asarray(
        [stoi[c] for c in text],
        dtype=np.int64,
    )

    train = torch.from_numpy(
        encoded[:train_end]
    )

    return train, len(chars)


def get_batch(source, generator):
    max_start = source.numel() - BLOCK - 1

    ix = torch.randint(
        0,
        max_start,
        (BATCH,),
        generator=generator,
    )

    offsets = torch.arange(
        BLOCK + 1,
        dtype=torch.long,
    )

    positions = (
        ix[:, None]
        + offsets[None, :]
    )

    batch = source[positions]

    return (
        batch[:, :-1],
        batch[:, 1:],
    )


# ============================================================
# CHECKPOINT
# ============================================================

def build_model_and_optimizer(vocab_size):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint.get(
        "config",
        {},
    )

    expected = {
        "block": BLOCK,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "n_emb": N_EMB,
        "batch": BATCH,
        "weight_decay": WD,
        "grad_clip": GRAD_CLIP,
        "adam_eps": ADAM_EPS,
        "vocab": vocab_size,
    }

    for key, value in expected.items():
        actual = config.get(key)

        if actual != value:
            raise RuntimeError(
                f"Checkpoint mismatch for {key}: "
                f"expected {value!r}, got {actual!r}"
            )

    model = Glyph(vocab_size)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()
    # Training dropout is required. The benchmark starts from the same
    # checkpoint but performs actual training steps.
    model.train()

    decay = []
    no_decay = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    optimizer = torch.optim.AdamW(
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

    # Load the exact optimizer state from the 20k checkpoint, then keep the
    # continuation LR fixed at 1e-6 just like the 50k experiment.
    optimizer.load_state_dict(
        checkpoint["opt"]
    )

    for group in optimizer.param_groups:
        group["lr"] = LR

    return model, optimizer


def train_step(
    model,
    optimizer,
    x,
    y,
):
    optimizer.zero_grad(
        set_to_none=True
    )

    loss = model(
        x,
        y,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    optimizer.step()

    return loss


# ============================================================
# TIMING
# ============================================================

def run_eager(
    threads,
    train_tensor,
    vocab_size,
    warmup,
    steps,
):
    torch.set_num_threads(
        threads
    )

    # Reset the same checkpoint/optimizer for every configuration.
    model, optimizer = (
        build_model_and_optimizer(
            vocab_size
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(12345)

    # Warmup.
    model.train()

    for _ in range(warmup):
        x, y = get_batch(
            train_tensor,
            generator,
        )
        train_step(
            model,
            optimizer,
            x,
            y,
        )

    # Timed region.
    times = []

    for _ in range(steps):
        x, y = get_batch(
            train_tensor,
            generator,
        )

        t0 = time.perf_counter()

        train_step(
            model,
            optimizer,
            x,
            y,
        )

        times.append(
            time.perf_counter()
            - t0
        )

    total = sum(times)
    mean_step = statistics.mean(times)
    median_step = statistics.median(times)

    return {
        "mode": "eager",
        "threads": threads,
        "compile_sec": 0.0,
        "mean_step_sec": mean_step,
        "median_step_sec": median_step,
        "steps_per_sec": 1.0 / mean_step,
        "chars_per_sec": (
            BATCH * BLOCK / mean_step
        ),
        "projected_30k_min": (
            30_000 * mean_step / 60.0
        ),
        "projected_30k_hours": (
            30_000 * mean_step / 3600.0
        ),
        "total_benchmark_sec": total,
    }


def run_compiled(
    threads,
    train_tensor,
    vocab_size,
    warmup,
    steps,
):
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "This PyTorch build does not provide torch.compile()."
        )

    torch.set_num_threads(
        threads
    )

    model, optimizer = (
        build_model_and_optimizer(
            vocab_size
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(12345)

    compile_start = time.perf_counter()

    compiled_model = torch.compile(
        model,
        backend="inductor",
        mode="max-autotune",
    )

    # Compilation usually happens on first execution, so warmup includes the
    # compilation cost. We keep it separate from the timed steady-state.
    compiled_model.train()

    for _ in range(warmup):
        x, y = get_batch(
            train_tensor,
            generator,
        )

        train_step(
            compiled_model,
            optimizer,
            x,
            y,
        )

    compile_sec = (
        time.perf_counter()
        - compile_start
    )

    times = []

    for _ in range(steps):
        x, y = get_batch(
            train_tensor,
            generator,
        )

        t0 = time.perf_counter()

        train_step(
            compiled_model,
            optimizer,
            x,
            y,
        )

        times.append(
            time.perf_counter()
            - t0
        )

    total = sum(times)
    mean_step = statistics.mean(times)
    median_step = statistics.median(times)

    return {
        "mode": "torch.compile",
        "threads": threads,
        "compile_sec": compile_sec,
        "mean_step_sec": mean_step,
        "median_step_sec": median_step,
        "steps_per_sec": 1.0 / mean_step,
        "chars_per_sec": (
            BATCH * BLOCK / mean_step
        ),
        "projected_30k_min": (
            30_000 * mean_step / 60.0
        ),
        "projected_30k_hours": (
            30_000 * mean_step / 3600.0
        ),
        "total_benchmark_sec": total,
    }


def print_result(result):
    print()
    print(
        f"{result['mode']:14s} | "
        f"threads={result['threads']:2d} | "
        f"step={result['mean_step_sec']:.4f}s | "
        f"{result['steps_per_sec']:.3f} steps/s | "
        f"{result['chars_per_sec']:.0f} chars/s | "
        f"30k={result['projected_30k_hours']:.2f}h"
    )

    if result["mode"] == "torch.compile":
        print(
            f"                 compile/warmup={result['compile_sec']:.1f}s"
        )


def main():
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    available = os.cpu_count() or 1

    requested_threads = []
    for part in args.threads.split(","):
        value = int(part.strip())
        if value <= 0:
            continue
        if value <= available:
            requested_threads.append(value)

    requested_threads = list(
        dict.fromkeys(requested_threads)
    )

    if not requested_threads:
        raise RuntimeError(
            f"No requested thread counts are <= available CPU count ({available})."
        )

    print("=" * 78)
    print("Glyph v2 CPU SPEED BENCHMARK")
    print("=" * 78)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CPU logical     : {available}")
    print(f"Warmup steps    : {args.warmup}")
    print(f"Timed steps     : {args.steps}")
    print(f"Batch           : {BATCH}")
    print(f"Context         : {BLOCK}")
    print("Checkpoint      : glyph_v2.pt (read only)")
    print()

    train_tensor, vocab_size = load_data()

    print(
        f"Training chars : {train_tensor.numel():,}"
    )
    print(
        f"Vocabulary     : {vocab_size}"
    )
    print()

    results = []

    # First: eager thread scaling.
    print(
        "EAGER THREAD TEST"
    )
    print("-" * 78)

    for threads in requested_threads:
        try:
            result = run_eager(
                threads,
                train_tensor,
                vocab_size,
                args.warmup,
                args.steps,
            )
            results.append(result)
            print_result(result)

        except Exception as exc:
            print()
            print(
                f"EAGER threads={threads} FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

    if not results:
        raise RuntimeError(
            "All eager speed tests failed."
        )

    fastest_eager = min(
        results,
        key=lambda r: r["mean_step_sec"],
    )

    print()
    print("=" * 78)
    print("FASTEST EAGER CONFIG")
    print("=" * 78)
    print_result(
        fastest_eager
    )

    # Second: compile only the best eager thread count.
    compile_result = None

    print()
    print(
        "TORCH.COMPILE TEST"
    )
    print("-" * 78)
    print(
        "Compiling only the fastest eager thread count to keep this benchmark short."
    )

    try:
        compile_result = run_compiled(
            fastest_eager["threads"],
            train_tensor,
            vocab_size,
            args.warmup,
            args.steps,
        )
        results.append(
            compile_result
        )
        print_result(
            compile_result
        )

    except Exception as exc:
        print()
        print(
            f"torch.compile FAILED: "
            f"{type(exc).__name__}: {exc}"
        )

    # Final comparison.
    print()
    print("=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)

    results_sorted = sorted(
        results,
        key=lambda r: r["mean_step_sec"],
    )

    for rank, result in enumerate(
        results_sorted,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['mode']} "
            f"threads={result['threads']}: "
            f"{result['mean_step_sec']:.4f}s/step, "
            f"{result['projected_30k_hours']:.2f}h for 30k steps"
        )

    print()
    best = results_sorted[0]

    current_hours = 30_000 * 0.522 / 3600.0
    best_hours = best[
        "projected_30k_hours"
    ]

    if current_hours > 0:
        speedup = (
            current_hours / best_hours
        )
    else:
        speedup = float("nan")

    print(
        f"Fastest benchmark projection: "
        f"{best_hours:.2f} hours for 30k steps"
    )

    print(
        f"Approx. speedup vs the observed "
        f"~0.522 s/step run: "
        f"{speedup:.2f}x"
    )

    print()
    print(
        "IMPORTANT: This script only benchmarks speed."
    )
    print(
        "It never writes a training checkpoint and does not alter glyph_v2.pt."
    )
    print(
        "Use the fastest CONFIGURATION reported here for the real 20k -> 50k run."
    )


if __name__ == "__main__":
    main()
