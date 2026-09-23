"""Interactive chat / text continuation for Glyph v3-A."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT_DIR / "models/v3/glyph_v3_lstm_best.pt"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW_TOKENS = 500


class GlyphV3LSTM(nn.Module):
    def __init__(self, vocab_size: int, emb: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, emb)
        self.rnn = nn.LSTM(
            input_size=emb,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, hidden=None):
        x = self.tok(idx)
        x, hidden = self.rnn(x, hidden)
        return self.head(x), hidden


def sample_next(logits, temperature, top_k):
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered[indices] = values
        logits = filtered
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.inference_mode()
def generate(model, prompt, stoi, itos, device, seq_len, max_new_tokens, temperature, top_k):
    prompt = prompt.lower() or " "
    unknown = sorted(set(ch for ch in prompt if ch not in stoi))
    if unknown:
        raise ValueError("Unknown characters: " + " ".join(repr(ch) for ch in unknown))

    ids = [stoi[ch] for ch in prompt][-seq_len:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # The prompt is evaluated once. The last prompt-position logits predict
    # the first generated character; the new character is then fed back for
    # the following step. This avoids processing the last prompt character twice.
    logits, hidden = model(idx)
    generated = []

    for _ in range(max_new_tokens):
        next_id = sample_next(logits[0, -1], temperature, top_k)
        generated.append(itos[next_id])
        current = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden = model(current, hidden)

    return "".join(generated)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    chars = config["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    model = GlyphV3LSTM(
        vocab_size=len(chars),
        emb=int(config["emb"]),
        hidden=int(config["hidden"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("=" * 64)
    print("Glyph v3-A Interactive Chat")
    print("=" * 64)
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Step         : {checkpoint.get('step', '?')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', float('nan')):.6f}")
    print(f"Vocabulary   : {len(chars)}")
    print(f"Embedding    : {config['emb']}")
    print(f"Hidden       : {config['hidden']}")
    print(f"Layers       : {config['layers']}")
    print(f"Context      : {config['seq_len']}")
    print(f"Device       : {device}")
    print(f"Temperature  : {args.temperature}")
    print(f"Top-k        : {args.top_k}")
    print(f"New chars    : {args.max_new_tokens}")
    print("=" * 64)
    print("V3-A is a character-level continuation model. Use :quit to exit.")

    temperature = args.temperature
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens

    while True:
        try:
            prompt = input("\nYou: ").strip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt == ":quit":
            print("Goodbye!")
            break
        if prompt.startswith(":temp "):
            temperature = float(prompt.split(maxsplit=1)[1])
            print(f"Temperature set to {temperature}")
            continue
        if prompt.startswith(":topk "):
            top_k = int(prompt.split(maxsplit=1)[1])
            print(f"Top-k set to {top_k}")
            continue
        if prompt.startswith(":tokens "):
            max_new_tokens = int(prompt.split(maxsplit=1)[1])
            print(f"New chars set to {max_new_tokens}")
            continue

        try:
            completion = generate(
                model,
                prompt,
                stoi,
                itos,
                device,
                int(config["seq_len"]),
                max_new_tokens,
                temperature,
                top_k,
            )
            print(f"Glyph: {prompt.lower()}{completion}")
        except ValueError as exc:
            print(f"Glyph: {exc}")


if __name__ == "__main__":
    main()
