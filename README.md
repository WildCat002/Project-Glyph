# Project Glyph

Glyph is a small character-level language-model project built to study how a language model learns text generation from raw characters.

The project deliberately avoids pretrained language models, external AI APIs, tokenizer libraries, and instruction tuning. The goal is to keep the implementation small enough to inspect, modify, train, and benchmark directly.

## Current release

**Glyph v2.0.0 — controlled Transformer baseline**

V2 introduced a proper held-out validation split, a larger context window, a deeper Transformer, dedicated checkpoints, and a broader evaluation suite. The successful extended run reached 100,000 training steps on a CUDA T4.

The best 100k checkpoint is:

```text
models/v2/glyph_v2_100k_cuda_best.pt
```

Its best validation checkpoint occurred at step 99,000.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the released V2 model interactively:

```bash
python src/v2/glyph_v2_chat.py
```

The default chat settings are:

```text
checkpoint    models/v2/glyph_v2_100k_cuda_best.pt
context       128 characters
temperature   0.7
top-k         20
new chars     500
```

The interactive chat accepts:

```text
:temp 0.9
:topk 40
:tokens 300
:quit
```

Glyph is a text-continuation model, not an instruction-tuned assistant. A prompt is treated as context and the model continues it character by character.

## Repository structure

The repository is organized by role and model generation rather than keeping every artifact in the root directory.

```text
Glyph/
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   └── data.txt
│
├── models/
│   ├── v1/
│   │   └── glyph.pt
│   └── v2/
│       ├── glyph_v2.pt
│       ├── glyph_v2_best.pt
│       ├── glyph_v2_50k_best.pt
│       └── glyph_v2_100k_cuda_best.pt
│
├── src/
│   ├── v1/
│   │   ├── glyph.py
│   │   ├── glyph_chat.py
│   │   └── benchmark.py
│   └── v2/
│       ├── glyph_v2.py
│       ├── glyph_v2_chat.py
│       ├── benchmark_v2.py
│       ├── speed_benchmark_v2.py
│       ├── glyph_v2_loss_anatomy.py
│       ├── glyph_v2_generation_dynamics.py
│       └── glyph_v2_generation_dynamics_v2.py
│
├── experiments/
│   ├── v2_20k/
│   │   ├── glyph_v2_history.csv
│   │   └── glyph_v2_last_good.pt
│   ├── v2_50k/
│   │   ├── glyph_v2_50k.py
│   │   ├── glyph_v2_50k.pt
│   │   ├── glyph_v2_50k_last_good.pt
│   │   └── glyph_v2_50k_history.csv
│   └── v2_100k/
│       ├── glyph_v2_100k.py
│       ├── glyph_v2_100k_cuda.py
│       ├── glyph_v2_100k_cuda.pt
│       ├── glyph_v2_100k_cuda_last_good.pt
│       └── glyph_v2_100k_cuda_history.csv
│
└── results/
    ├── v1/
    │   └── benchmark_results.txt
    └── v2/
        ├── benchmark_v2_results.txt
        ├── glyph_v2_generation_dynamics_fixed_results.*
        ├── glyph_v2_generation_dynamics_v2_results.*
        └── glyph_v2_loss_anatomy_results.*
```

The root is intentionally small. `models/` contains the checkpoints that are useful as named baselines. `experiments/` contains continuation checkpoints and training histories. `results/` contains evaluation outputs.

## V1 baseline

V1 is preserved as the historical baseline from `v1.0.0`.

Train:

```bash
python src/v1/glyph.py
```

Chat:

```bash
python src/v1/glyph_chat.py
```

Benchmark:

```bash
python src/v1/benchmark.py
```

V1 model:

```text
models/v1/glyph.pt
```

## V2 architecture

Glyph V2 uses a decoder-only character-level Transformer.

```text
Vocabulary      98 characters
Context         128 characters
Layers          4
Attention heads 4
Embedding       128
Batch size      32
Parameters      820,224
```

Training data uses a deterministic split of the 5,000,000-character corpus:

```text
Training       first 4,750,000 characters
Validation     final 250,000 characters
```

The vocabulary is reconstructed with the same sorted character set used during training. The same corpus must therefore be used with the included checkpoints.

## V2 training

Base 20k training:

```bash
python src/v2/glyph_v2.py
```

The base run writes its main checkpoints to:

```text
models/v2/glyph_v2.pt
models/v2/glyph_v2_best.pt
```

Recovery state and training history are kept under:

```text
experiments/v2_20k/
```

### 20k -> 50k

```bash
python experiments/v2_50k/glyph_v2_50k.py
```

### 50k -> 100k CUDA

The successful extended run used a CUDA T4:

```bash
python experiments/v2_100k/glyph_v2_100k_cuda.py
```

The selected best 100k checkpoint is promoted to:

```text
models/v2/glyph_v2_100k_cuda_best.pt
```

The raw continuation checkpoint, recovery checkpoint, and history remain under:

```text
experiments/v2_100k/
```

## Benchmarks

Base V2 benchmark:

```bash
python src/v2/benchmark_v2.py
```

Loss anatomy:

```bash
python src/v2/glyph_v2_loss_anatomy.py
```

This examines overall loss/perplexity, top-k accuracy, letter prediction, whitespace prediction, punctuation prediction, and space precision/recall/F1.

Generation dynamics:

```bash
python src/v2/glyph_v2_generation_dynamics_v2.py
```

The generation-dynamics study uses real held-out validation anchors and compares greedy decoding with top-k sampling at multiple temperatures.

Speed benchmark:

```bash
python src/v2/speed_benchmark_v2.py
```

Evaluation outputs are stored under `results/v1/` and `results/v2/`.

## V2 results at a glance

The extended V2 training run improved teacher-forced validation performance substantially:

```text
20k best validation loss    ~2.3607
50k best validation loss    ~2.2292
100k best validation loss   ~1.9100
```

The 100k loss-anatomy benchmark also showed large improvements in next-character prediction and whitespace prediction compared with the 20k checkpoint.

Free-running generation remained a separate problem. Greedy decoding could become highly repetitive, while top-k sampling reduced that failure mode. For this reason the project does not treat a single metric as a complete measure of generation quality.

## Data

The included corpus is the 5,000,000-character training corpus used by the V2 runs.

To build a different corpus, replace `data/data.txt` and rebuild the character vocabulary through a fresh training run. Existing checkpoints may no longer match a changed character vocabulary.

For example, a new Gutenberg-based corpus can be assembled into `data/data.txt` and then cleaned before training.

## Reproducibility

The main experiments use deterministic seeds where practical and record training configuration in checkpoints and CSV histories.

When adding a new model generation, keep its checkpoint, training script, benchmark scripts, and results separated from previous releases so earlier baselines remain reproducible.

## Design principles

Glyph intentionally avoids:

- pretrained language models
- external AI APIs
- tokenizer libraries
- instruction tuning
- large-model infrastructure

The project is primarily a learning and experimentation environment: build the model, train it, inspect what it learns, benchmark it, and keep the experiment history.

## Version history

```text
v1.0.0  first trained Transformer baseline
v2.0.0  controlled Transformer baseline with held-out validation and extended training
```
