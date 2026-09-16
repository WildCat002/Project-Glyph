# Glyph

A small character-level decoder-only Transformer language model, built from scratch.

Glyph is a personal learning and experimentation project focused on understanding how language models work by building and training one from the ground up.

## What it is

Glyph predicts the next character given the characters before it.

The current Glyph v1 model uses:

- Architecture: decoder-only Transformer
- Tokenizer: raw characters
- Vocabulary: 98 characters for the v1 training corpus
- Context length: 64 characters
- Layers: 3
- Attention heads: 4
- Embedding dimension: 128
- Parameters: 614,272
- Training framework: PyTorch
- Training device: CPU
- Training data: 5,000,000 characters
- Maximum training steps: 20,000

Glyph is a small experimental language model. It is not a production model or an instruction-following/conversational AI system.

## Project History

Glyph started as a simple NumPy neural network and evolved into the Transformer implementation in this version.

The original implementation is kept in Git history to document the project's progression.

The current Transformer implementation uses:

- Multi-head causal self-attention
- Layer normalization
- GELU activation
- Residual connections
- Weight tying between token embeddings and the output head
- AdamW optimization
- Learning-rate warmup
- Cosine learning-rate decay
- Gradient clipping
- Checkpointing

The project intentionally avoids starting from an existing pretrained language model.

## Features

- No pretrained weights
- No AI APIs
- Character-level tokenization
- Decoder-only Transformer
- Trained from the provided text corpus
- CPU-based training
- Inference using the trained checkpoint
- Benchmarking and generation tests
- Checkpoint resume support

## Install

### Linux / macOS

```bash
pip install -r requirements.txt
```

### Windows

```powershell
pip install -r requirements.txt
```

### Termux / Android

PyTorch should be installed through the Termux package manager rather than pip.

```bash
pkg update
pkg install python python-torch python-numpy
```

Then verify PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

Do not use `pip install torch` on Termux.

## Get Training Data

Glyph can be trained on a large plain-text corpus.

Project Gutenberg is one possible source of public-domain books. The following example downloads several books into one file:

```bash
rm -f data.txt

for id in 11 1661 2701 1342 84 98 345 2600 1400 1260 1080 158 174 120; do
  curl -sL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" >> data.txt
  echo "" >> data.txt
done

wc -c data.txt
```

You can then remove the Gutenberg headers/footers and prepare the corpus:

```bash
python - <<'EOF'
import re

t = open('data.txt', 'r', encoding='utf-8', errors='ignore').read()
t = re.sub(r'\*\*\* ?START OF.*?\*\*\*', '', t, flags=re.S)
t = re.sub(r'\*\*\* ?END OF.*?\*\*\*', '', t, flags=re.S)
t = t.lower()[:5_000_000]

open('data.txt', 'w', encoding='utf-8').write(t)
print(len(t), "chars")
EOF
```

The training script currently uses at most the first 5 million characters.

More and better training data can improve results, but training time also increases.

## Train

Run:

```bash
python glyph.py
```

Training is checkpointed to:

```text
glyph.pt
```

A checkpoint is saved every 500 steps. If training is interrupted, running the script again can resume from the last saved checkpoint.

For long CPU training sessions, `tmux` can be useful:

```bash
tmux new -s glyph
python glyph.py
```

Detach with:

```text
Ctrl+B
D
```

Reattach later with:

```bash
tmux attach -t glyph
```

## Generate Text

After training:

```bash
python glyph_chat.py
```

Glyph loads `glyph.pt` and continues a prompt using patterns learned from the training corpus.

Example:

```text
Loaded Glyph @ step 20000
Glyph ready. 'quit' to exit.

> alice was
```

Type:

```text
quit
```

to exit.

This is text generation, not a conversational AI system. Glyph has not been instruction-tuned or specifically trained for dialogue.

### Sampling Parameters

The `generate()` function supports:

- `n` — number of characters to generate
- `temp` — sampling temperature
- `top_k` — limits sampling to the top-k predictions

Lower temperatures generally produce more predictable output, while higher temperatures produce more varied output.

## Benchmark

Glyph v1 includes a benchmark script for checking the trained checkpoint and measuring basic generation behavior.

Run:

```bash
python benchmark.py
```

The benchmark checks:

- Checkpoint integrity
- Model parameter count
- Vocabulary size
- Evaluation loss
- Perplexity
- Generation at multiple temperatures
- Repetition statistics
- A simple corpus-overlap/memorization signal

Results are written to:

```text
benchmark_results.txt
```

### Glyph v1 Benchmark Results

The completed v1 training run reached step 20,000 with a healthy checkpoint.

```text
Checkpoint step : 20000
Parameters      : 614,272
Vocabulary      : 98
Corpus chars    : 5,000,000

Checkpoint finite: YES

Evaluation loss : 2.2765
Perplexity      : 9.74
```

The evaluation uses the final 100,000 characters of the 5,000,000-character corpus. Because v1 was trained on the first 5,000,000 characters, this is a training-overlap evaluation rather than a true held-out validation set.

The simple memorization check found:

```text
0/10 20-char chunks found in corpus
```

for each of the seven benchmark prompts, for a total of:

```text
0/70
```

This is only a basic signal and is not a definitive memorization test.

## Training

Training speed and loss depend on the dataset, hardware, PyTorch version, and random initialization.

The completed Glyph v1 training run used:

```text
Training steps : 20,000
Peak LR        : 1e-5
Warmup         : 4,000 steps
Gradient clip  : 0.05
Weight decay   : 0.01
```

The final training step reached:

```text
step 20000
loss 2.2675
```

Actual training time can vary significantly between devices.

## Files

```text
glyph/
├── .gitignore
├── README.md
├── data.txt
├── glyph.py
├── glyph_chat.py
├── benchmark.py
├── benchmark_results.txt
├── glyph.pt
└── requirements.txt
```

`data.txt` is included in the repository as the training corpus used by this version of Glyph.

`glyph.pt` is the trained Glyph v1 checkpoint and is included so the model can be used immediately after cloning without retraining.

`benchmark_results.txt` records the benchmark results for the included v1 checkpoint.

## Tuning

The main training parameters are near the top of `glyph.py`:

```python
BLOCK = 64
N_LAYER = 3
N_HEAD = 4
N_EMB = 128
BATCH = 32
MAX_STEPS = 20000
LR = 1e-5
```

If training becomes unstable, possible experiments include:

- Lowering `LR`
- Increasing `WARMUP`
- Lowering the gradient clipping threshold
- Increasing AdamW `eps`

To increase model capacity, you can experiment with:

- More layers
- A larger embedding dimension
- More attention heads
- A longer context window
- More training steps
- More training data

Increasing model size and context length also increases training cost.

## Why Glyph?

The goal of Glyph is to learn by building the system rather than starting with an existing language model.

Even though it is a small project, it covers several important concepts behind modern language models:

- Tokenization
- Embeddings
- Self-attention
- Transformer blocks
- Residual connections
- Layer normalization
- Autoregressive training
- Sampling
- Optimization
- Checkpointing

## Status

**Glyph v1.0.0 — First trained baseline**

The first complete Glyph version has been trained for 20,000 steps and includes a runnable trained checkpoint and benchmark results.

The project remains an ongoing personal learning and experimentation project. Future versions may change the architecture, training procedure, dataset, context length, and model capacity significantly.
