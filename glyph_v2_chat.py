"""
Glyph v2 Interactive Chat / Text Generation
===========================================

Load the released Glyph v2 100k best checkpoint and interact with it from
an interactive terminal.

Glyph is a character-level language model, not an instruction-tuned chatbot.
Each prompt is treated as a text continuation. The model generates characters
that statistically follow the supplied context.

Default checkpoint:
    glyph_v2_100k_cuda_best.pt

The checkpoint was trained on the same data.txt used by Glyph v2. The
character vocabulary is reconstructed from that file, so data.txt must be
present and must match the training corpus.

Examples:
    python glyph_v2_chat.py
    python glyph_v2_chat.py --temperature 0.7 --top-k 20 --max-new-tokens 300
    python glyph_v2_chat.py --checkpoint glyph_v2.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# V2 MODEL CONFIG
# ============================================================

BLOCK = 128
N_LAYER = 4
N_HEAD = 4
N_EMB = 128
DROPOUT = 0.1

DEFAULT_CHECKPOINT = "glyph_v2_100k_cuda_best.pt"
DATA_FILE = "data.txt"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW_TOKENS = 500


# ============================================================
# MODEL
# ============================================================


class Attn(nn.Module):
    def __init__(self):
        super().__init__()

        self.qkv = nn.Linear(N_EMB, 3 * N_EMB, bias=False)
        self.proj = nn.Linear(N_EMB, N_EMB, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        q, k, v = self.qkv(x).split(N_EMB, dim=2)
        head_dim = channels // N_HEAD

        q = q.view(batch_size, seq_len, N_HEAD, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, N_HEAD, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, N_HEAD, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
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
    def __init__(self, vocab_size: int):
        super().__init__()

        self.tok = nn.Embedding(vocab_size, N_EMB)
        self.pos = nn.Embedding(BLOCK, N_EMB)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMB)
        self.head = nn.Linear(N_EMB, vocab_size, bias=False)

        # Recreate the training architecture. The checkpoint already contains
        # the trained weights, so initialization only matters for construction.
        self.head.weight = self.tok.weight

    def forward(self, idx):
        _, seq_len = idx.shape

        if seq_len > BLOCK:
            raise ValueError(f"Sequence length {seq_len} exceeds BLOCK={BLOCK}")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.tok(idx) + self.pos(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================
# CHECKPOINT / TOKENIZER
# ============================================================


def load_vocabulary(data_path: Path):
    try:
        text = data_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError as exc:
        raise SystemExit(f"Could not read {data_path}: {exc}") from exc

    if not text:
        raise SystemExit(f"{data_path} is empty.")

    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}
    return text, stoi, itos


def load_checkpoint(checkpoint_path: Path, device: torch.device, vocab_size: int):
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Place the checkpoint in the project directory or pass "
            f"--checkpoint PATH."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise SystemExit(
            f"{checkpoint_path} does not look like a Glyph v2 checkpoint."
        )

    config = checkpoint.get("config", {})
    checkpoint_vocab = config.get("vocab")

    if checkpoint_vocab is not None and int(checkpoint_vocab) != vocab_size:
        raise SystemExit(
            "Vocabulary mismatch:\n"
            f"  checkpoint: {checkpoint_vocab}\n"
            f"  data.txt:   {vocab_size}\n"
            "Use the same data.txt that was used during training."
        )

    model = Glyph(vocab_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, checkpoint


# ============================================================
# GENERATION
# ============================================================


def sample_next(logits: torch.Tensor, temperature: float, top_k: int):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered[indices] = values
        logits = filtered

    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).item()


@torch.inference_mode()
def generate(
    model: nn.Module,
    prompt: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    if not prompt:
        prompt = " "

    # Glyph v2 was trained on a lowercased corpus, so normalize prompts the
    # same way before tokenization.
    prompt = prompt.lower()

    unknown = sorted(set(char for char in prompt if char not in stoi))
    if unknown:
        readable = " ".join(repr(char) for char in unknown)
        raise ValueError(
            f"Prompt contains characters that are not in the training vocabulary: {readable}"
        )

    prompt_ids = [stoi[char] for char in prompt]

    # The Transformer can only see the last BLOCK characters. Keeping the tail
    # is the same behavior used by normal autoregressive generation.
    prompt_ids = prompt_ids[-BLOCK:]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_chars = []

    for _ in range(max_new_tokens):
        context = idx[:, -BLOCK:]
        logits = model(context)[:, -1, :].squeeze(0)
        next_id = sample_next(logits, temperature, top_k)

        idx = torch.cat(
            [idx, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        generated_chars.append(itos[next_id])

    return "".join(generated_chars)


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively generate text with the Glyph v2 model."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Keep only the top K logits; 0 disables filtering (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Characters to generate per prompt (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when CUDA is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.temperature <= 0:
        raise SystemExit("--temperature must be greater than 0.")

    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0.")

    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be greater than 0.")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    data_path = Path(DATA_FILE)
    checkpoint_path = Path(args.checkpoint)

    _, stoi, itos = load_vocabulary(data_path)
    model, checkpoint = load_checkpoint(
        checkpoint_path,
        device,
        len(stoi),
    )

    params = sum(parameter.numel() for parameter in model.parameters())
    step = checkpoint.get("step", "unknown")
    best_val = checkpoint.get("best_val_loss")

    print("=" * 64)
    print("Glyph v2 Interactive Chat")
    print("=" * 64)
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Step         : {step}")
    if isinstance(best_val, (float, int)) and math.isfinite(float(best_val)):
        print(f"Best val loss: {float(best_val):.6f}")
    print(f"Parameters   : {params:,}")
    print(f"Vocabulary   : {len(stoi)}")
    print(f"Context      : {BLOCK}")
    print(f"Device       : {device}")
    print(f"Temperature  : {args.temperature}")
    print(f"Top-k        : {args.top_k}")
    print(f"New chars    : {args.max_new_tokens}")
    print("=" * 64)
    print("Enter a prompt and Glyph will continue it.")
    print("Commands: :quit, :temp VALUE, :topk VALUE, :tokens VALUE")
    print()

    temperature = args.temperature
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens

    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.strip().lower() in {":quit", ":q", "quit", "exit"}:
            print("Goodbye!")
            break

        command = prompt.strip()

        if command.lower().startswith(":temp "):
            try:
                temperature = float(command.split(maxsplit=1)[1])
                if temperature <= 0:
                    raise ValueError
                print(f"Temperature set to {temperature}")
            except ValueError:
                print("Usage: :temp 0.7")
            continue

        if command.lower().startswith(":topk "):
            try:
                top_k = int(command.split(maxsplit=1)[1])
                if top_k < 0:
                    raise ValueError
                print(f"Top-k set to {top_k}")
            except ValueError:
                print("Usage: :topk 20")
            continue

        if command.lower().startswith(":tokens "):
            try:
                max_new_tokens = int(command.split(maxsplit=1)[1])
                if max_new_tokens <= 0:
                    raise ValueError
                print(f"Generation length set to {max_new_tokens}")
            except ValueError:
                print("Usage: :tokens 500")
            continue

        try:
            continuation = generate(
                model=model,
                prompt=prompt,
                stoi=stoi,
                itos=itos,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            continue
        except RuntimeError as exc:
            print(f"Generation error: {exc}")
            continue

        print(f"Glyph: {prompt}{continuation}")
        print()


if __name__ == "__main__":
    main()
