#!/usr/bin/env python3

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

# This file:
# Glyph/src/v3/glyph_v3d_chat.py
#
# Project root:
# Glyph/

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = ROOT / "models" / "v3" / "glyph_v3d.pt"


# ============================================================
# DEFAULT SAMPLING SETTINGS
# ============================================================

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_MAX_CHARS = 500


# ============================================================
# V3-D MODEL
# ============================================================


class GlyphV3LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int = 96,
        hidden_size: int = 240,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # IMPORTANT:
        # These names must match the trained V3-D checkpoint.

        self.tok = nn.Embedding(
            vocab_size,
            emb_size,
        )

        self.rnn = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.head = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(
        self,
        idx,
        hidden=None,
    ):
        x = self.tok(idx)

        x, hidden = self.rnn(
            x,
            hidden,
        )

        logits = self.head(x)

        return logits, hidden


# ============================================================
# CHECKPOINT TYPE
# ============================================================


def get_checkpoint_type(path: Path) -> str:
    name = path.stem.lower()

    if "best_task" in name:
        return "task-best"

    if "best_combined" in name:
        return "combined-best"

    if name == "glyph_v3d":
        return "final-50k"

    return "custom"


# ============================================================
# LOAD CHECKPOINT
# ============================================================


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n" f"{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model'.")

    config = checkpoint.get(
        "config",
        {},
    )

    chars = config.get("chars")

    if not chars:
        raise ValueError("Checkpoint does not contain " "config['chars'].")

    # Positional arguments intentionally used here
    # so the constructor cannot suffer from keyword
    # mismatches.

    model = GlyphV3LSTM(
        len(chars),
        int(config.get("emb", 96)),
        int(config.get("hidden", 240)),
        int(config.get("layers", 2)),
        float(config.get("dropout", 0.1)),
    )

    # Must exactly match the V3-D checkpoint.
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.to(device)
    model.eval()

    stoi = {c: i for i, c in enumerate(chars)}

    itos = {i: c for i, c in enumerate(chars)}

    seq_len = int(
        config.get(
            "seq_len",
            128,
        )
    )

    return (
        model,
        stoi,
        itos,
        seq_len,
        config,
        checkpoint,
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
    temperature=0.7,
    top_k=20,
    max_new_chars=500,
    seed=None,
):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    if max_new_chars <= 0:
        raise ValueError("max_new_chars must be greater than 0")

    if seed is not None:
        torch.manual_seed(seed)

    # Character-level tokenization.
    ids = [stoi[c] for c in prompt.lower() if c in stoi]

    if not ids:
        raise ValueError("Prompt contains no characters " "known by the model.")

    device = next(model.parameters()).device

    # Keep only the supported context window.
    ids = ids[-seq_len:]

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    # Process the prompt once.
    logits, hidden = model(x)

    generated = []

    for _ in range(max_new_chars):

        # Last-token logits.
        z = logits[0, -1].float()

        # Temperature.
        z = z / temperature

        # Top-k sampling.
        if top_k > 0:

            k = min(
                top_k,
                z.numel(),
            )

            values, indices = torch.topk(
                z,
                k=k,
            )

            probs = F.softmax(
                values,
                dim=-1,
            )

            next_id = indices[
                torch.multinomial(
                    probs,
                    1,
                )
            ].item()

        else:

            probs = F.softmax(
                z,
                dim=-1,
            )

            next_id = torch.multinomial(
                probs,
                1,
            ).item()

        generated.append(itos[next_id])

        # Feed only the newly generated character.
        x = torch.tensor(
            [[next_id]],
            dtype=torch.long,
            device=device,
        )

        logits, hidden = model(
            x,
            hidden,
        )

    return "".join(generated)


# ============================================================
# OUTPUT CLEANING
# ============================================================


def clean_output(text: str) -> str:
    """
    Stop if the model starts generating another synthetic
    user turn.
    """

    text = re.split(
        r"\n\s*user\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    return text.strip()


# ============================================================
# PARAMETER COUNT
# ============================================================


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# CHECKPOINT INFO
# ============================================================


def show_info(
    checkpoint_path,
    model,
    config,
    checkpoint,
    seq_len,
    vocab_size,
    temperature,
    top_k,
    max_chars,
    device,
):
    params = count_parameters(model)

    checkpoint_type = get_checkpoint_type(checkpoint_path)

    print()

    print("=" * 64)
    print("Glyph V3-D Checkpoint Info")
    print("=" * 64)

    print(f"Checkpoint Type : " f"{checkpoint_type}")

    print(f"Checkpoint      : " f"{checkpoint_path}")

    print(f"Step            : " f"{checkpoint.get('step', 'not stored')}")

    print(f"Parameters      : " f"{params:,}")

    print(f"Context         : " f"{seq_len}")

    print(f"Vocab           : " f"{vocab_size}")

    print(f"Embedding       : " f"{config.get('emb', 96)}")

    print(f"Hidden          : " f"{config.get('hidden', 240)}")

    print(f"Layers          : " f"{config.get('layers', 2)}")

    print(f"Dropout         : " f"{config.get('dropout', 0.1)}")

    print(f"Device          : " f"{device}")

    print()

    print(f"Temperature     : " f"{temperature}")

    print(f"Top-k           : " f"{top_k}")

    print(f"Max chars       : " f"{max_chars}")

    print()

    # Different V3-D checkpoint files may store
    # different metadata keys. Show whatever exists.

    print(f"Best val loss   : " f"{checkpoint.get('best_val_loss', 'not stored')}")

    print(f"Best combined   : " f"{checkpoint.get('best_combined', 'not stored')}")

    print(f"Best task       : " f"{checkpoint.get('best_task', 'not stored')}")

    print(f"Best natural    : " f"{checkpoint.get('best_natural_val', 'not stored')}")

    print("=" * 64)
    print()


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(description="Glyph V3-D interactive chat")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to V3-D checkpoint",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k sampling",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum generated characters",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    if args.topk < 0:
        raise ValueError("--topk must be >= 0")

    if args.max_chars <= 0:
        raise ValueError("--max-chars must be greater than 0")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    (
        model,
        stoi,
        itos,
        seq_len,
        config,
        checkpoint,
    ) = load_checkpoint(
        args.checkpoint,
        device,
    )

    temperature = args.temperature
    top_k = args.topk
    max_chars = args.max_chars

    history = ""

    params = count_parameters(model)

    # --------------------------------------------------------
    # Startup display
    # --------------------------------------------------------

    print()

    print("=" * 64)
    print("Glyph V3-D Chat")
    print("=" * 64)

    print(f"Checkpoint : " f"{args.checkpoint}")

    print(f"Type       : " f"{get_checkpoint_type(args.checkpoint)}")

    print(f"Step       : " f"{checkpoint.get('step', '?')}")

    print(f"Parameters : " f"{params:,}")

    print(f"Context    : " f"{seq_len}")

    print(f"Vocab      : " f"{len(stoi)}")

    print(f"Device     : " f"{device}")

    print(f"Temperature: " f"{temperature}")

    print(f"Top-k      : " f"{top_k}")

    print(f"Max chars  : " f"{max_chars}")

    print()

    print("Commands:")
    print("  /reset       reset conversation")
    print("  /temp 0.7    change temperature")
    print("  /topk 20     change top-k")
    print("  /max 500     change output length")
    print("  /info        show checkpoint information")
    print("  /quit        exit")

    print("=" * 64)

    # --------------------------------------------------------
    # Chat loop
    # --------------------------------------------------------

    while True:

        try:

            user_text = input("\nYou: ").strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print("\nBye.")
            break

        if not user_text:
            continue

        command = user_text.lower()

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if command in {
            "/quit",
            "/exit",
        }:

            print("Bye.")
            break

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        if command == "/reset":

            history = ""

            print("Conversation reset.")

            continue

        # ----------------------------------------------------
        # Info
        # ----------------------------------------------------

        if command == "/info":

            show_info(
                args.checkpoint,
                model,
                config,
                checkpoint,
                seq_len,
                len(stoi),
                temperature,
                top_k,
                max_chars,
                device,
            )

            continue

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        if command.startswith("/temp "):

            try:

                value = float(user_text.split(maxsplit=1)[1])

                if value <= 0:
                    raise ValueError

                temperature = value

                print(f"Temperature set to " f"{temperature}.")

            except ValueError:

                print("Usage: /temp 0.7")

            continue

        # ----------------------------------------------------
        # Top-k
        # ----------------------------------------------------

        if command.startswith("/topk "):

            try:

                value = int(user_text.split(maxsplit=1)[1])

                if value < 0:
                    raise ValueError

                top_k = value

                print(f"Top-k set to " f"{top_k}.")

            except ValueError:

                print("Usage: /topk 20")

            continue

        # ----------------------------------------------------
        # Max chars
        # ----------------------------------------------------

        if command.startswith("/max "):

            try:

                value = int(user_text.split(maxsplit=1)[1])

                if value <= 0:
                    raise ValueError

                max_chars = value

                print(f"Max chars set to " f"{max_chars}.")

            except ValueError:

                print("Usage: /max 500")

            continue

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        if history:

            prompt = history + f"user: {user_text}\n" + "assistant:"

        else:

            prompt = f"user: {user_text}\n" "assistant:"

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            reply = generate(
                model=model,
                stoi=stoi,
                itos=itos,
                seq_len=seq_len,
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                max_new_chars=max_chars,
                seed=args.seed,
            )

            reply = clean_output(reply)

        except Exception as exc:

            print(f"Generation error: {exc}")

            continue

        if not reply:
            reply = "(no output)"

        print(f"Glyph: {reply}")

        # ----------------------------------------------------
        # Conversation history
        # ----------------------------------------------------

        history += f"user: {user_text}\n" f"assistant: {reply}\n"

        # Keep history bounded.
        history = history[-4096:]


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
