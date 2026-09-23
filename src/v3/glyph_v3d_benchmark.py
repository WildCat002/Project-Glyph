from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = ROOT / "models/v3/glyph_v3d.pt"

DEFAULT_OUTPUT = ROOT / "results/v3/benchmark_v3D_results.txt"


# ============================================================
# BENCHMARK CONFIG
# ============================================================

TEMPERATURE = 0.7
TOP_K = 20
NEW_CHARS = 120
SEEDS = [42, 43, 44, 45, 46]


# ============================================================
# SAME TASK SET AS V3-C
# ============================================================

TASKS = [
    {
        "id": 1,
        "category": "arithmetic",
        "prompt": "user: what is 7 + 8?\nassistant:",
        "expected": "15",
    },
    {
        "id": 2,
        "category": "arithmetic",
        "prompt": "user: what is 80 - 4?\nassistant:",
        "expected": "76",
    },
    {
        "id": 3,
        "category": "arithmetic",
        "prompt": "user: what is 12 times 4?\nassistant:",
        "expected": "48",
    },
    {
        "id": 4,
        "category": "arithmetic",
        "prompt": "user: what is 81 divided by 9?\nassistant:",
        "expected": "9",
    },
    {
        "id": 5,
        "category": "comparison",
        "prompt": "user: which is larger, 30 or 87?\nassistant:",
        "expected": "87",
    },
    {
        "id": 6,
        "category": "comparison",
        "prompt": "user: which is smaller, 23 or 76?\nassistant:",
        "expected": "23",
    },
    {
        "id": 7,
        "category": "sequence",
        "prompt": "user: what comes next: 2, 5, 8, 11, ?\nassistant:",
        "expected": "14",
    },
    {
        "id": 8,
        "category": "sequence",
        "prompt": "user: what comes next: 10, 16, 22, 28, ?\nassistant:",
        "expected": "34",
    },
    {
        "id": 9,
        "category": "yes_no",
        "prompt": "user: is 17 greater than 9?\nassistant:",
        "expected": "yes",
    },
    {
        "id": 10,
        "category": "yes_no",
        "prompt": "user: is 4 less than 2?\nassistant:",
        "expected": "no",
    },
    {
        "id": 11,
        "category": "string",
        "prompt": "user: reverse the word stone.\nassistant:",
        "expected": "enots",
    },
    {
        "id": 12,
        "category": "string",
        "prompt": "user: how many letters are in apple?\nassistant:",
        "expected": "5",
    },
    {
        "id": 13,
        "category": "logic",
        "prompt": "user: all zibs are blue. alice is a zib. what color is alice?\nassistant:",
        "expected": "blue",
    },
    {
        "id": 14,
        "category": "logic",
        "prompt": "user: alice is taller than bob. bob is taller than carol. who is tallest?\nassistant:",
        "expected": "alice",
    },
    {
        "id": 15,
        "category": "logic",
        "prompt": "user: bob arrives before carol. carol arrives before david. who arrives first?\nassistant:",
        "expected": "bob",
    },
    {
        "id": 16,
        "category": "conversation",
        "prompt": "user: hello\nassistant:",
        "expected": "hello",
        "acceptable": ["hello", "hi", "hey", "hi there", "hello there", "greetings"],
    },
    {
        "id": 17,
        "category": "conversation",
        "prompt": "user: thank you\nassistant:",
        "expected": "happy",
        "acceptable": [
            "happy",
            "happy to help",
            "glad to help",
            "you're welcome",
            "you are welcome",
            "welcome",
            "my pleasure",
            "no problem",
            "anytime",
            "of course",
            "sure thing",
        ],
    },
    {
        "id": 18,
        "category": "conversation",
        "prompt": "user: good bye\nassistant:",
        "expected": "goodbye",
        "acceptable": [
            "goodbye",
            "good bye",
            "bye",
            "see you",
            "take care",
            "farewell",
        ],
    },
]


# ============================================================
# MODEL
# ============================================================


class GlyphV3LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb,
        hidden,
        layers,
        dropout,
    ):
        super().__init__()

        self.tok = nn.Embedding(
            vocab_size,
            emb,
        )

        self.rnn = nn.LSTM(
            emb,
            hidden,
            layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
        )

        self.head = nn.Linear(
            hidden,
            vocab_size,
        )

    def forward(
        self,
        idx,
        hidden=None,
    ):
        x, hidden = self.rnn(
            self.tok(idx),
            hidden,
        )

        return self.head(x), hidden


# ============================================================
# CHECKPOINT
# ============================================================


def load_checkpoint(path):
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ck["config"]

    chars = cfg["chars"]

    stoi = {c: i for i, c in enumerate(chars)}

    itos = {i: c for c, i in stoi.items()}

    model = GlyphV3LSTM(
        len(chars),
        int(cfg.get("emb", 96)),
        int(cfg.get("hidden", 240)),
        int(cfg.get("layers", 2)),
        float(cfg.get("dropout", 0.1)),
    )

    model.load_state_dict(ck["model"])

    model.eval()

    return (
        model,
        stoi,
        itos,
        int(cfg.get("seq_len", 128)),
        ck,
    )


# ============================================================
# GENERATION
# ============================================================


@torch.inference_mode()
def generate(
    model,
    stoi,
    itos,
    seq_len,
    prompt,
    seed,
):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ids = [stoi[c] for c in prompt.lower() if c in stoi]

    if not ids:
        raise ValueError(f"No known prompt characters for {prompt!r}")

    device = next(model.parameters()).device

    ids = ids[-seq_len:]

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    logits, hidden = model(idx)

    generated = []

    for _ in range(NEW_CHARS):
        z = logits[0, -1].float() / TEMPERATURE

        k = min(
            TOP_K,
            z.numel(),
        )

        values, indices = torch.topk(
            z,
            k,
        )

        probs = F.softmax(
            values,
            dim=-1,
        )

        nxt = indices[
            torch.multinomial(
                probs,
                1,
            )
        ].item()

        generated.append(itos[nxt])

        x = torch.tensor(
            [[nxt]],
            dtype=torch.long,
            device=device,
        )

        logits, hidden = model(
            x,
            hidden,
        )

    return "".join(generated)


# ============================================================
# ANSWER PARSING
# ============================================================


def first_answer(text):
    # Stop if model fabricates a new user turn.
    text = re.split(
        r"\n\s*user\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    for line in text.splitlines():
        line = line.strip()

        if line:
            return line

    return ""


def norm(text):
    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s'-]",
        "",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# SCORING
# ============================================================


def hit(category, expected, answer, acceptable=None):
    a = norm(answer)
    e = norm(expected)

    if category in {
        "arithmetic",
        "comparison",
        "sequence",
    }:
        m = re.search(r"(?<!\d)-?\d+(?!\d)", a)
        return bool(m and m.group() == e)

    if category == "yes_no":
        return bool(a.split()) and a.split()[0] == e

    if category == "string":
        if e.isdigit():
            m = re.search(r"\d+", a)
            return bool(m and m.group() == e)
        return a.startswith(e)

    if category == "logic":
        return e in a[:100]

    if category == "conversation":
        options = [norm(x) for x in (acceptable or [expected])]
        return any(option and option in a[:100] for option in options)

    return False


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Glyph V3-D task benchmark with semantic conversation scoring"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="V3-D checkpoint to benchmark.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output report path.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU.",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: " f"{args.checkpoint}")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    (
        model,
        stoi,
        itos,
        seq_len,
        ck,
    ) = load_checkpoint(args.checkpoint)

    model.to(device)

    params = sum(p.numel() for p in model.parameters())

    report = [
        "Glyph V3-D Task Benchmark",
        "=" * 68,
        f"Checkpoint   : {args.checkpoint.resolve()}",
        f"Step         : {ck.get('step', 'unknown')}",
        f"Best val loss: {ck.get('best_val_loss', 'not stored')}",
        f"Best combined: {ck.get('best_combined', 'not stored')}",
        f"Best task    : {ck.get('best_task', 'not stored')}",
        f"Best natural : {ck.get('best_natural_val', 'not stored')}",
        f"Parameters   : {params:,}",
        f"Device       : {device}",
        f"Temperature  : {TEMPERATURE}",
        f"Top-k        : {TOP_K}",
        f"New chars    : {NEW_CHARS}",
        f"Seeds        : {SEEDS}",
        f"Tasks        : {len(TASKS)}",
        "Conversation scoring: accepted-response sets (not one exact phrase).",
        "",
    ]

    total_hits = 0
    total_cases = 0

    category_hits = {}
    category_cases = {}

    for number, task in enumerate(TASKS, 1):
        category = task["category"]
        prompt = task["prompt"]
        expected = task["expected"]
        acceptable = task.get("acceptable")

        task_hits = 0

        for seed in SEEDS:

            answer = first_answer(
                generate(
                    model,
                    stoi,
                    itos,
                    seq_len,
                    prompt,
                    seed + number * 1000,
                )
            )

            ok = hit(
                category,
                expected,
                answer,
                acceptable,
            )

            total_hits += int(ok)
            total_cases += 1
            task_hits += int(ok)

            category_hits[category] = category_hits.get(
                category,
                0,
            ) + int(ok)

            category_cases[category] = (
                category_cases.get(
                    category,
                    0,
                )
                + 1
            )

            report.extend(
                [
                    (
                        f"TASK {number:02d} | "
                        f"{category:12s} | "
                        f"seed={seed} | "
                        f"{'PASS' if ok else 'FAIL'}"
                    ),
                    f"prompt:   {prompt!r}",
                    f"expected: {expected!r}",
                    (
                        f"acceptable: {acceptable!r}"
                        if acceptable
                        else "acceptable: [exact / category rule]"
                    ),
                    f"answer:   {answer!r}",
                    "",
                ]
            )

        report.append(
            f"TASK {number:02d} accuracy: "
            f"{task_hits}/{len(SEEDS)} = "
            f"{task_hits / len(SEEDS):.4f}"
        )

        report.append("")

    report.extend(
        [
            "=" * 68,
            "CATEGORY RESULTS",
            "=" * 68,
        ]
    )

    for category in category_hits:
        hits = category_hits[category]
        cases = category_cases[category]

        report.append(f"{category:14s}: " f"{hits}/{cases} = " f"{hits / cases:.4f}")

    report.extend(
        [
            "",
            "=" * 68,
            ("Expected-answer hit rate: " f"{total_hits / total_cases:.4f}"),
            f"Total correct: {total_hits}/{total_cases}",
            "=" * 68,
            "",
            "Method notes:",
            "1. Every task is sampled with five deterministic seeds.",
            "2. Only the first non-empty assistant answer is scored.",
            "3. Fabricated later user/assistant turns are excluded.",
            "4. Arithmetic/comparison/sequence require the expected number.",
            "5. Yes/no requires the expected first token.",
            "6. String tasks require the expected exact prefix/value.",
            "7. Logic tasks use a bounded expected-answer match.",
            "8. Conversation tasks use an acceptable-response set; the old single target is shown only as a reference.",
        ]
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("=" * 68)
    print("Benchmark complete. " f"Results saved to: " f"{args.output.resolve()}")
    print("Expected-answer hit rate: " f"{total_hits / total_cases:.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
