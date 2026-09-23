from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ============================================================
# CONFIG
# ============================================================

SEED = 42
CPU_THREADS = 4

DATA_PATH = Path("data.txt")

MAX_DATA = 5_000_000
VAL_CHARS = 250_000
TRAIN_NATURAL_CHARS = MAX_DATA - VAL_CHARS

SEQ_LEN = 128
BATCH = 32

EMB = 96
HIDDEN = 240
LAYERS = 2
DROPOUT = 0.10

MAX_STEPS = 50_000
WARMUP = 1_000

MAX_LR = 2e-3
MIN_LR = 2e-4

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

TASK_TRAIN_EXAMPLES = 20_000
TASK_VAL_EXAMPLES = 5_000

TASK_TRAIN_SEED = 1337
TASK_VAL_SEED = 7331

SAVE_EVERY = 500
EVAL_EVERY = 500
LOG_EVERY = 100

CKPT = Path("glyph_v3d.pt")
BEST_COMBINED = Path("glyph_v3d_best_combined.pt")
BEST_TASK = Path("glyph_v3d_best_task.pt")
HISTORY = Path("glyph_v3d_history.csv")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(CPU_THREADS)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("Glyph V3-D")
print("=" * 72)
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: " f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing {DATA_PATH}")

text = DATA_PATH.read_text(
    encoding="utf-8",
    errors="ignore",
).lower()[:MAX_DATA]

if len(text) < MAX_DATA:
    print(f"Warning: data.txt contains only " f"{len(text):,} characters.")

val_start = max(
    len(text) - VAL_CHARS,
    TRAIN_NATURAL_CHARS,
)

natural_train = text[:val_start]
natural_val = text[val_start:]

chars = sorted(set(text))
V = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

print(f"Natural chars: {len(text):,}")
print(f"Natural train: {len(natural_train):,}")
print(f"Natural val:   {len(natural_val):,}")
print(f"Vocabulary:    {V}")


# ============================================================
# SYNTHETIC TASK GENERATION
# ============================================================

TRAIN_NAMES = [
    "alice",
    "bob",
    "carol",
    "david",
    "emma",
    "frank",
    "grace",
    "henry",
]

VAL_NAMES = [
    "irene",
    "jack",
    "kate",
    "leo",
    "maria",
    "nina",
    "oscar",
    "paul",
]

TRAIN_WORDS = [
    "apple",
    "house",
    "train",
    "blue",
    "stone",
    "river",
    "plant",
    "book",
    "chair",
    "table",
    "water",
]

VAL_WORDS = [
    "garden",
    "window",
    "planet",
    "orange",
    "silver",
    "rabbit",
    "morning",
    "machine",
    "forest",
    "teacher",
]

TRAIN_COLORS = [
    "blue",
    "green",
    "red",
]

VAL_COLORS = [
    "yellow",
    "purple",
    "orange",
]


def arithmetic_task(
    rng: random.Random,
    validation: bool,
):
    if validation:
        mode = rng.choice(
            [
                "add",
                "sub",
                "mul",
                "div",
            ]
        )
        low = 70
        high = 99
    else:
        mode = rng.choice(
            [
                "add",
                "sub",
                "mul",
                "div",
            ]
        )
        low = 10
        high = 69

    if mode == "add":
        a = rng.randint(low, high)
        b = rng.randint(low, high)

        return (
            f"user: what is {a} + {b}?\n" f"assistant:",
            str(a + b),
        )

    if mode == "sub":
        a = rng.randint(low, high)
        b = rng.randint(low, a)

        return (
            f"user: what is {a} - {b}?\n" f"assistant:",
            str(a - b),
        )

    if mode == "mul":
        if validation:
            a = rng.randint(10, 19)
            b = rng.randint(3, 9)
        else:
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)

        return (
            f"user: what is {a} times {b}?\n" f"assistant:",
            str(a * b),
        )

    if validation:
        divisor = rng.randint(3, 9)
        quotient = rng.randint(10, 19)
    else:
        divisor = rng.randint(2, 9)
        quotient = rng.randint(2, 10)

    dividend = divisor * quotient

    return (
        f"user: what is {dividend} divided by {divisor}?\n" f"assistant:",
        str(quotient),
    )


def comparison_task(
    rng: random.Random,
    validation: bool,
):
    low = 70 if validation else 10
    high = 99 if validation else 69

    a = rng.randint(low, high)
    b = rng.randint(low, high)

    if a == b:
        b = a - 1

    if rng.random() < 0.5:
        if a < b:
            answer = a
        else:
            answer = b

        return (
            f"user: which is smaller, {a} or {b}?\n" f"assistant:",
            str(answer),
        )

    if a > b:
        answer = a
    else:
        answer = b

    return (
        f"user: which is larger, {a} or {b}?\n" f"assistant:",
        str(answer),
    )


def sequence_task(
    rng: random.Random,
    validation: bool,
):
    if validation:
        start = rng.randint(30, 60)
        step = rng.randint(11, 18)
    else:
        start = rng.randint(1, 30)
        step = rng.randint(2, 9)

    values = [start + step * i for i in range(4)]

    answer = values[-1] + step

    sequence = ", ".join(str(x) for x in values)

    return (
        f"user: what comes next: {sequence}, ?\n" f"assistant:",
        str(answer),
    )


def yes_no_task(
    rng: random.Random,
    validation: bool,
):
    low = 70 if validation else 10
    high = 99 if validation else 69

    a = rng.randint(low, high)
    b = rng.randint(low, high)

    if rng.random() < 0.5:
        return (
            f"user: is {a} greater than {b}?\n" f"assistant:",
            "yes" if a > b else "no",
        )

    return (
        f"user: is {a} less than {b}?\n" f"assistant:",
        "yes" if a < b else "no",
    )


def string_task(
    rng: random.Random,
    validation: bool,
):
    words = VAL_WORDS if validation else TRAIN_WORDS
    word = rng.choice(words)

    mode = rng.choice(
        [
            "length",
            "reverse",
            "spell",
        ]
    )

    if mode == "length":
        return (
            f"user: how many letters are in {word}?\n" f"assistant:",
            str(len(word)),
        )

    if mode == "reverse":
        return (
            f"user: reverse the word {word}.\n" f"assistant:",
            word[::-1],
        )

    return (
        f"user: spell the word {word} one letter at a time.\n" f"assistant:",
        " ".join(word),
    )


def logic_task(
    rng: random.Random,
    validation: bool,
):
    names = VAL_NAMES if validation else TRAIN_NAMES

    a, b, c = rng.sample(names, 3)

    mode = rng.choice(
        [
            "tall",
            "arrival",
            "color",
        ]
    )

    if mode == "tall":
        return (
            f"user: {a} is taller than {b}. "
            f"{b} is taller than {c}. "
            f"who is tallest?\n"
            f"assistant:",
            a,
        )

    if mode == "arrival":
        return (
            f"user: {a} arrives before {b}. "
            f"{b} arrives before {c}. "
            f"who arrives first?\n"
            f"assistant:",
            a,
        )

    color = rng.choice(VAL_COLORS if validation else TRAIN_COLORS)

    return (
        f"user: all zibs are {color}. "
        f"{a} is a zib. "
        f"what color is {a}?\n"
        f"assistant:",
        color,
    )


def conversation_task(
    rng: random.Random,
    validation: bool,
):
    if validation:
        options = [
            ("hello there", "hello"),
            ("thanks a lot", "welcome"),
            ("see you later", "goodbye"),
        ]
    else:
        options = [
            ("hello", "hello"),
            ("hi", "hello"),
            ("thank you", "welcome"),
            ("thanks", "welcome"),
            ("goodbye", "goodbye"),
            ("good bye", "goodbye"),
        ]

    prompt, answer = rng.choice(options)

    return (
        f"user: {prompt}\n" f"assistant:",
        answer,
    )


TASK_BUILDERS = [
    ("arithmetic", arithmetic_task, 30),
    ("comparison", comparison_task, 15),
    ("sequence", sequence_task, 15),
    ("yes_no", yes_no_task, 10),
    ("string", string_task, 15),
    ("logic", logic_task, 10),
    ("conversation", conversation_task, 5),
]


def build_task_text(
    count: int,
    seed: int,
    validation: bool,
):
    rng = random.Random(seed)

    expanded = []

    for name, fn, weight in TASK_BUILDERS:
        expanded.extend([(name, fn)] * weight)

    records = []

    counts = {name: 0 for name, _, _ in TASK_BUILDERS}

    for _ in range(count):
        name, fn = rng.choice(expanded)

        prompt, answer = fn(
            rng,
            validation,
        )

        records.append(f"{prompt}{answer}\n\n")

        counts[name] += 1

    return "".join(records), counts


task_train_text, train_counts = build_task_text(
    TASK_TRAIN_EXAMPLES,
    TASK_TRAIN_SEED,
    False,
)

task_val_text, val_counts = build_task_text(
    TASK_VAL_EXAMPLES,
    TASK_VAL_SEED,
    True,
)


print()
print("Synthetic task training:")
print(f"  chars: {len(task_train_text):,}")
print(f"  counts: {train_counts}")

print("Synthetic task validation:")
print(f"  chars: {len(task_val_text):,}")
print(f"  counts: {val_counts}")


# ============================================================
# VOCABULARY SAFETY
# ============================================================

allowed = set(chars)

extra_chars = sorted(set(task_train_text + task_val_text) - allowed)

if extra_chars:
    raise ValueError(
        "Synthetic task data contains characters "
        "missing from natural-text vocabulary:\n" + repr(extra_chars)
    )


# ============================================================
# TOKENIZATION
# ============================================================


def encode(s: str):
    return np.asarray(
        [stoi[c] for c in s],
        dtype=np.int64,
    )


train_natural = encode(natural_train)
val_natural = encode(natural_val)

train_tasks = encode(task_train_text)
val_tasks = encode(task_val_text)

# Natural corpus stays intact; task data is added as a second
# training stream. Random sampling makes the effective task
# proportion equal to its character share.
train_np = np.concatenate(
    [
        train_natural,
        train_tasks,
    ]
)

print()
print(f"Train stream: {len(train_np):,} chars")
print(f"Train tasks:  {len(train_tasks):,} chars")
print(f"Task share:   {len(train_tasks) / len(train_np):.3f}")


# ============================================================
# TORCH TENSORS
# ============================================================

train_data = torch.from_numpy(train_np).to(
    DEVICE,
    dtype=torch.long,
)

natural_val_data = torch.from_numpy(val_natural).to(
    DEVICE,
    dtype=torch.long,
)

task_val_data = torch.from_numpy(val_tasks).to(
    DEVICE,
    dtype=torch.long,
)


# ============================================================
# MODEL
# ============================================================


class GlyphV3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok = nn.Embedding(
            V,
            EMB,
        )

        self.rnn = nn.LSTM(
            input_size=EMB,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            dropout=(DROPOUT if LAYERS > 1 else 0.0),
        )

        self.head = nn.Linear(
            HIDDEN,
            V,
        )

    def forward(
        self,
        idx,
    ):
        x = self.tok(idx)

        x, _ = self.rnn(x)

        logits = self.head(x)

        return logits


model = GlyphV3D().to(DEVICE)

params = sum(p.numel() for p in model.parameters())

print()
print(f"Parameters: {params:,}")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=MAX_LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# BATCHING
# ============================================================

offsets = torch.arange(
    SEQ_LEN + 1,
    device=DEVICE,
)


def sample_batch(data):
    max_start = data.numel() - SEQ_LEN - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (BATCH,),
        device=DEVICE,
    )

    positions = starts[:, None] + offsets[None, :]

    batch = data[positions]

    return (
        batch[:, :-1],
        batch[:, 1:],
    )


@torch.no_grad()
def evaluate(
    data,
    batches=64,
):
    model.eval()

    losses = []

    for _ in range(batches):
        x, y = sample_batch(data)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, V).float(),
            y.reshape(-1),
        )

        losses.append(loss.item())

    model.train()

    return float(np.mean(losses))


# ============================================================
# LR SCHEDULE
# ============================================================


def current_lr(step):
    if step <= WARMUP:
        return MAX_LR * step / max(WARMUP, 1)

    progress = (step - WARMUP) / max(
        MAX_STEPS - WARMUP,
        1,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def set_lr(lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


# ============================================================
# CHECKPOINT
# ============================================================

config = {
    "version": "v3-d",
    "vocab": V,
    "chars": chars,
    "seq_len": SEQ_LEN,
    "emb": EMB,
    "hidden": HIDDEN,
    "layers": LAYERS,
    "dropout": DROPOUT,
    "batch": BATCH,
    "max_steps": MAX_STEPS,
    "warmup": WARMUP,
    "max_lr": MAX_LR,
    "min_lr": MIN_LR,
    "weight_decay": WEIGHT_DECAY,
    "grad_clip": GRAD_CLIP,
    "natural_train_chars": len(natural_train),
    "natural_val_chars": len(natural_val),
    "task_train_examples": TASK_TRAIN_EXAMPLES,
    "task_val_examples": TASK_VAL_EXAMPLES,
    "task_train_seed": TASK_TRAIN_SEED,
    "task_val_seed": TASK_VAL_SEED,
}


def save_checkpoint(
    path,
    step,
    best_nat,
    best_task,
    best_combined,
):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "step": step,
            "best_nat_val_loss": best_nat,
            "best_task_val_loss": best_task,
            "best_combined": best_combined,
            "config": config,
        },
        path,
    )


# ============================================================
# TRAIN
# ============================================================

history_exists = HISTORY.exists()

with HISTORY.open(
    "a",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    if not history_exists:
        writer.writerow(
            [
                "step",
                "train_loss",
                "natural_val_loss",
                "task_val_loss",
                "combined_val_loss",
                "lr",
                "grad_norm",
            ]
        )

    best_nat = float("inf")
    best_task = float("inf")
    best_combined = float("inf")

    running_losses = []

    start_time = time.time()

    model.train()

    for step in range(
        1,
        MAX_STEPS + 1,
    ):
        lr = current_lr(step)
        set_lr(lr)

        x, y = sample_batch(train_data)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, V).float(),
            y.reshape(-1),
        )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: " f"{loss.item()}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")

        optimizer.step()

        running_losses.append(loss.item())

        if step % LOG_EVERY == 0:
            avg = float(np.mean(running_losses))

            running_losses.clear()

            elapsed = time.time() - start_time

            print(
                f"step {step:6d} | "
                f"train {avg:.4f} | "
                f"lr {lr:.3e} | "
                f"grad {float(grad_norm):.4f} | "
                f"time {elapsed/60:.1f}m"
            )

        if step % EVAL_EVERY == 0 or step == 1:
            natural_loss = evaluate(
                natural_val_data,
                batches=64,
            )

            task_loss = evaluate(
                task_val_data,
                batches=64,
            )

            combined = 0.5 * natural_loss + 0.5 * task_loss

            print(
                f"           "
                f"natural_val {natural_loss:.4f} | "
                f"task_val {task_loss:.4f} | "
                f"combined {combined:.4f}"
            )

            writer.writerow(
                [
                    step,
                    avg if "avg" in locals() else float("nan"),
                    natural_loss,
                    task_loss,
                    combined,
                    lr,
                    float(grad_norm),
                ]
            )

            f.flush()

            best_nat = min(best_nat, natural_loss)

            if task_loss < best_task:
                best_task = task_loss

                save_checkpoint(
                    BEST_TASK,
                    step,
                    best_nat,
                    best_task,
                    best_combined,
                )

                print(f"           " f"saved {BEST_TASK.name}")

            if combined < best_combined:
                best_combined = combined

                save_checkpoint(
                    BEST_COMBINED,
                    step,
                    best_nat,
                    best_task,
                    best_combined,
                )

                print(f"           " f"saved {BEST_COMBINED.name}")

        if step % SAVE_EVERY == 0:
            save_checkpoint(
                CKPT,
                step,
                best_nat,
                best_task,
                best_combined,
            )

            print(f"           " f"saved {CKPT.name}")


# ============================================================
# FINAL
# ============================================================

elapsed = time.time() - start_time

save_checkpoint(
    CKPT,
    MAX_STEPS,
    best_nat,
    best_task,
    best_combined,
)

print()
print("=" * 72)
print("V3-D TRAINING COMPLETE")
print("=" * 72)
print(f"steps:              {MAX_STEPS:,}")
print(f"parameters:         {params:,}")
print(f"best natural val:   {best_nat:.6f}")
print(f"best task val:      {best_task:.6f}")
print(f"best combined:      {best_combined:.6f}")
print(f"time:               {elapsed / 60:.1f} minutes")
print()
print(f"checkpoint:         {CKPT}")
print(f"best combined:      {BEST_COMBINED}")
print(f"best task:          {BEST_TASK}")
print(f"history:            {HISTORY}")
