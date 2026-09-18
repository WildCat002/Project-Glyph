# Glyph

A small character-level language model built from scratch in PyTorch.

Glyph is a personal learning and experimentation project focused on understanding how language models work by implementing, training, evaluating, and iterating on the architecture rather than starting from a pretrained model.

Glyph is intentionally small. It is not a production language model, an instruction-tuned assistant, or a general-purpose chatbot.

## What Glyph does

Glyph learns to predict the next character from the characters before it.

The project is character-level rather than word- or subword-level, so the model receives individual character IDs and learns spelling, whitespace, punctuation, and longer text patterns directly from sequences of characters.

## Version history

### v1.0.0 — Transformer baseline

The first complete Glyph version established the baseline:

- Decoder-only Transformer
- Character-level vocabulary
- 98-character vocabulary
- Context length: 64
- 3 Transformer layers
- 4 attention heads
- Embedding size: 128
- 614,272 parameters
- 5,000,000-character corpus
- 20,000 training steps

The v1 implementation and checkpoint are preserved in the repository and in the `v1.0.0` Git tag.

The v1 interactive generator is:

```bash
python glyph_chat.py
```

### v2.0.0 — Controlled Transformer experiment

Glyph v2 keeps the same character-level approach while making the evaluation setup more rigorous and increasing model/context capacity.

| Setting | Glyph v2 |
|---|---:|
| Architecture | Decoder-only Transformer |
| Vocabulary | 98 characters |
| Context length | 128 |
| Layers | 4 |
| Attention heads | 4 |
| Embedding size | 128 |
| Parameters | 820,224 |
| Total corpus | 5,000,000 chars |
| Training split | 4,750,000 chars |
| Validation split | 250,000 chars |
| Base training run | 20,000 steps |
| Extended run | 100,000 steps |
| Framework | PyTorch |

The main v2 changes were:

- A real held-out validation split using the final 250,000 characters.
- A larger context window: 128 instead of 64.
- Four Transformer layers instead of three.
- Separate v2 checkpoints so the v1 model is never modified.
- Periodic validation and best-checkpoint tracking.
- Training history recorded to CSV.
- Controlled loss, quality, and generation-dynamics benchmarks.
- CUDA training support for extended runs.
- An interactive V2 generation script that loads the trained 100k checkpoint.

## Training results

The base v2 run completed at step 20,000 without numerical instability.

The same architecture was then continued to 50,000 and 100,000 steps on a Tesla T4.

Approximate validation results from those experiments:

| Checkpoint | Best validation loss | Approx. perplexity |
|---|---:|---:|
| v2 20k | 2.36 | 10.60 |
| v2 50k | 2.23 | 9.29 |
| v2 100k | 1.91 | 6.76 |

The 100k run reached its best training-time validation loss at step 99,000, then finished at step 100,000 with a slightly higher validation loss. The released extended checkpoint is therefore the 99k best-validation checkpoint:

```text
glyph_v2_100k_cuda_best.pt
```

These numbers are not directly comparable to the original v1 benchmark loss because v1's evaluation used a training-overlap region, while v2 uses a true held-out validation split.

The extended v2 model also produces more recognizable word-like text than the 20k checkpoint under controlled sampling, but long free-running generation can still become repetitive or malformed. Glyph remains an experimental character-level model rather than a general-purpose language model.

## Chat / text generation with Glyph v2

Glyph v2 is a text continuation model, not an instruction-tuned chatbot. A prompt is treated as context and the model generates characters that statistically follow that context.

The interactive V2 generator uses:

```text
glyph_v2_100k_cuda_best.pt
```

by default.

Run:

```bash
python glyph_v2_chat.py
```

Example:

```text
================================================================
Glyph v2 Interactive Chat
================================================================
Checkpoint   : glyph_v2_100k_cuda_best.pt
Step         : 99000
Best val loss: 1.909972
Parameters   : 820,224
Vocabulary   : 98
Context      : 128
Device       : cpu
Temperature  : 0.7
Top-k        : 20
New chars    : 500
================================================================
Enter a prompt and Glyph will continue it.
Commands: :quit, :temp VALUE, :topk VALUE, :tokens VALUE

You: Alice was
Glyph: Alice was ...
```

The default sampling configuration is:

```text
temperature = 0.7
top-k = 20
max new characters = 500
```

These values are practical defaults for the trained V2 checkpoint; they are not a guarantee of coherent output.

### Chat controls

Inside the interactive program:

```text
:temp 0.9      change temperature
:topk 40       change top-k
:tokens 300    change generation length
:quit          exit
```

You can also set them from the command line:

```bash
python glyph_v2_chat.py --temperature 0.7 --top-k 20 --max-new-tokens 300
```

To load another compatible V2 checkpoint:

```bash
python glyph_v2_chat.py --checkpoint glyph_v2.pt
```

To force CPU:

```bash
python glyph_v2_chat.py --cpu
```

The script automatically uses CUDA when available unless `--cpu` is supplied.

## Important checkpoint requirement

The V2 checkpoint stores the vocabulary size but not a standalone copy of the character-to-ID mapping.

The chat and benchmark programs therefore reconstruct the vocabulary with:

```python
chars = sorted(set(text))
```

from `data.txt`.

Use the same `data.txt` that was used during training. Changing the corpus can change the vocabulary ordering and make the checkpoint incompatible.

## Training data

The repository contains the 5,000,000-character corpus used by the included runs in `data.txt`.

The v2 split is:

```text
First 4,750,000 characters  -> training
Final 250,000 characters    -> validation
```

The vocabulary is reconstructed from the lowercase corpus.

### Building your own corpus

Glyph can also be trained on another plain-text corpus.

One possible source is Project Gutenberg. For example:

```bash
rm -f data.txt

for id in 11 1661 2701 1342 84 98 345 2600 1400 1260 1080 158 174 120; do
  curl -sL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" >> data.txt
  echo "" >> data.txt
done
```

You can then remove Gutenberg markers and prepare the corpus:

```bash
python - <<'PY'
import re

text = open("data.txt", "r", encoding="utf-8", errors="ignore").read()
text = re.sub(r"\*\*\* ?START OF.*?\*\*\*", "", text, flags=re.S)
text = re.sub(r"\*\*\* ?END OF.*?\*\*\*", "", text, flags=re.S)
text = text.lower()[:5_000_000]

open("data.txt", "w", encoding="utf-8").write(text)
print(len(text), "chars")
PY
```

Changing the corpus means an existing checkpoint may no longer be compatible with the reconstructed vocabulary.

## Install

### Windows / Linux / macOS

```bash
pip install -r requirements.txt
```

The project requires:

```text
torch>=2.0
numpy>=1.24
```

### Termux / Android

For Termux, install PyTorch through the Termux package rather than using `pip install torch`:

```bash
pkg update
pkg install python python-torch python-numpy
```

Verify the installation with:

```bash
python -c "import torch; print(torch.__version__)"
```

## Train Glyph v2

The main V2 training script is:

```bash
python glyph_v2.py
```

The default configuration trains from scratch for 20,000 steps.

Important settings are near the top of the script:

```python
BLOCK = 128
N_LAYER = 4
N_HEAD = 4
N_EMB = 128
BATCH = 32
MAX_STEPS = 20_000
WARMUP = 4_000
LR = 1e-5
WD = 0.01
GRAD_CLIP = 0.05
ADAM_EPS = 1e-6
```

The base run writes:

```text
glyph_v2.pt
glyph_v2_best.pt
glyph_v2_last_good.pt
glyph_v2_history.csv
```

The checkpoint contains the model state, optimizer state, training step, best validation loss, and training configuration.

## Extended V2 training

The repository also contains continuation scripts used for controlled experiments with the same architecture.

### 20k -> 50k

```bash
python glyph_v2_50k.py
```

### 50k -> 100k

For the successful CUDA run:

```bash
python glyph_v2_100k_cuda.py
```

The 100k experiment produced separate checkpoints so the original 20k V2 run remained intact.

These continuation scripts are experiment history, not different model architectures.

## Benchmarking

### Base V2 benchmark

```bash
python benchmark_v2.py
```

This evaluates the V2 checkpoint, runs generation tests at several temperatures, measures repetition, and performs a simple corpus-overlap signal.

### Loss anatomy

```bash
python glyph_v2_loss_anatomy.py
```

This breaks validation behavior into useful components such as:

- overall cross-entropy
- perplexity
- top-1 and top-5 accuracy
- letter prediction loss
- whitespace prediction loss
- punctuation prediction loss
- space precision / recall / F1

### Generation dynamics

```bash
python glyph_v2_generation_dynamics_v2.py
```

This uses deterministic held-out validation anchors and measures how generation behavior changes as the model runs farther away from the real context.

The benchmark compares greedy decoding with top-k sampling at several temperatures and reports how well the model's own free-running history stays aligned with the real held-out continuation.

## What the V2 experiments showed

V2 demonstrates why a single validation-loss number is not enough to describe a small language model.

As the model was trained longer, teacher-forced validation performance improved substantially. The 100k model is much better at next-character prediction, including letter and whitespace prediction, than the 20k model.

At the same time, free-running generation can become increasingly self-reinforcing. Greedy decoding in particular can produce highly repetitive text even when the generated characters form valid words. Top-k sampling reduces this failure mode, but long generations can still drift or repeat.

This is why the project uses multiple measurements rather than treating validation loss, word validity, or repetition as a complete quality score by itself.

## Project structure

The repository contains both the historical V1 baseline and the V2 experiments.

Core V2 files:

```text
README.md
requirements.txt
data.txt

glyph_v2.py
glyph_v2_chat.py
benchmark_v2.py
glyph_v2_loss_anatomy.py
glyph_v2_generation_dynamics_v2.py

glyph_v2.pt
glyph_v2_best.pt
glyph_v2_100k_cuda_best.pt
```

V1 files are preserved as historical baseline artifacts:

```text
glyph.py
glyph_chat.py
benchmark.py
glyph.pt
benchmark_results.txt
```

The repository also contains additional V2 continuation scripts, checkpoints, histories, and benchmark results from the experiments used to reach the 100k model.

## Design principles

Glyph intentionally avoids:

- Pretrained language models
- External AI APIs
- Tokenizer libraries
- Instruction tuning
- Large-model infrastructure

The goal is to keep the implementation understandable enough to inspect, modify, train, and benchmark directly.

## Why character-level?

Character-level modeling makes the mechanics visible.

The model does not receive a vocabulary of words or subwords. Every input is represented directly as a character ID. This keeps the implementation relatively small and makes the limitations of small language models visible: spelling, word formation, punctuation, whitespace, and long-range coherence all have to emerge from the character sequence alone.

## Status

**Glyph v2.0.0 — Controlled Transformer baseline with extended 100k training**

Glyph v2 established a cleaner held-out evaluation setup, increased the context/model capacity, and produced an extended 100k checkpoint with substantially lower validation loss than the 20k baseline.

The next major experiment is a controlled recurrent/LSTM architecture at approximately the same parameter budget. That experiment will be treated as a new model version rather than silently changing V2.
