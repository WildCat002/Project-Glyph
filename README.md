# Glyph

Glyph is a small-scale character-level language model project built as a
personal machine-learning research and learning project.

The project focuses on experimenting with language-model architectures,
training methods, evaluation, and model behavior at relatively small
parameter counts.

---

## Project Versions

| Version | Architecture | Description |
|---------|---------------|-------------|
| `v1.0.0` | Character Transformer | First trained Glyph baseline |
| `v2.0.0` | Controlled Character Transformer | Improved Transformer baseline with train/validation evaluation |
| `v2.0.1` | — | Repository organization and structure improvements |
| `v3.0.0` | Character LSTM | V3D baseline with task-oriented training and benchmarking |

---

# Glyph v3.0.0

`v3.0.0` introduces the **V3D** model, Glyph's third-generation
character-level language model.

V3D moves the project from the previous Transformer-based architecture to
a compact recurrent LSTM architecture and introduces a more structured
training and evaluation pipeline.

## V3D Architecture

```text
Character Input
      ↓
Embedding
      ↓
2-Layer LSTM
      ↓
Linear Output Head
      ↓
Character Probabilities
Model Configuration
Property	V3D
Vocabulary	98 characters
Context length	128
Embedding size	96
Hidden size	240
LSTM layers	2
Dropout	0.10
Parameters	820,226
Training steps	50,000


V3D is intentionally kept relatively small so that experiments can be
trained and evaluated on consumer hardware and used for learning and
research.
Training
V3D uses a natural-language corpus together with a synthetic task dataset.
Natural-Language Data
The corpus is split into training and validation sections:
Training:   first 4.75M characters
Validation: final 250K characters
Synthetic Task Training
V3D also includes structured synthetic tasks covering areas such as:
- Arithmetic
- Comparison
- Sequence completion
- Yes/No reasoning
- String manipulation
- Logic
- Conversation
The synthetic dataset used during V3D training contains approximately
998K task characters.
The purpose of the synthetic tasks is to give the small model explicit
examples of structured behaviors that are difficult to learn from
natural-language text alone.
V3D Checkpoints
The release contains three V3D checkpoints:
models/v3/
├── glyph_v3d.pt
├── glyph_v3d_best_task.pt
└── glyph_v3d_best_combined.pt
glyph_v3d.pt
The final V3D checkpoint from the 50,000-step training run.
glyph_v3d_best_task.pt
Checkpoint selected using task-oriented evaluation.
glyph_v3d_best_combined.pt
Checkpoint selected using the combined evaluation criterion used during
V3D experimentation.
Running V3D
The V3D chat interface is located at:
src/v3/glyph_v3d_chat.py
Run it from the repository root:
python src/v3/glyph_v3d_chat.py
The chat interface supports:
/reset
/temp
/topk
/max
/info
/quit
The /info command displays information about the loaded checkpoint,
model configuration, parameter count, sampling configuration, and
training step.
Benchmarking
V3 includes dedicated benchmarking tools for evaluating the model on
structured tasks and generation behavior.
The fixed V3D task benchmark contains 90 evaluation cases.
The final V3D checkpoint achieved:
48 / 90
53.33%
on the fixed task benchmark.
This benchmark is intended as a regression and task-behavior benchmark.
It should not be interpreted as a general measure of language-model
quality or general intelligence.
Future model versions can use the same benchmark to compare changes
against the V3D baseline.
Repository Structure
Glyph/
├── data/
│   └── v3/
│
├── experiments/
│   ├── v3_lstm/
│   ├── v3b/
│   ├── v3c/
│   └── v3d/
│
├── models/
│   └── v3/
│       ├── glyph_v3d.pt
│       ├── glyph_v3d_best_task.pt
│       └── glyph_v3d_best_combined.pt
│
├── results/
│   └── v3/
│
├── src/
│   └── v3/
│
├── README.md
└── requirements.txt
The experiments/ directories contain intermediate V3 research and
development work. They are kept in the repository to document the
development process, while the V3D checkpoints represent the official
v3.0.0 model line.
Development Philosophy
Glyph is intentionally a small research and learning project.
The goal is not to reproduce large production language models. Instead,
the project focuses on:
- Understanding neural-network architectures
- Training models from scratch
- Measuring model behavior
- Designing reproducible experiments
- Investigating generalization
- Learning from failures as well as successful experiments
Model versions are kept as explicit baselines so future experiments can
be compared against earlier results.
Experimental Work
The V3 development process includes several experimental branches and
architectural variations.
These experiments are preserved for research and comparison, but they
are not automatically considered part of the official V3D baseline.
Future experiments may investigate:
- Improved LSTM architectures
- Larger or deeper recurrent networks
- Better task and data diversity
- Improved generalization
- Alternative sampling strategies
- New Transformer architectures
- More rigorous unseen-task evaluation
Roadmap
Current baseline:
v3.0.0
  └── V3D LSTM baseline
Planned research directions:
V3D
 ├── V3D improvements
 ├── New training experiments
 └── Next-generation Transformer experiments
Future versions will be evaluated against the V3D baseline rather than
replacing its results.
Project History
Glyph has evolved through several controlled model generations:
v1.0.0
   ↓
First trained character-level Transformer

v2.0.0
   ↓
Controlled Transformer baseline

v2.0.1
   ↓
Repository organization

v3.0.0
   ↓
V3D character-level LSTM baseline
Each version is kept as part of the project's history so that changes in
architecture, training, and evaluation can be studied over time.
License
See the repository for the current license information.