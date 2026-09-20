import torch
import torch.nn.functional as F
from pathlib import Path

from torch import nn

torch.set_num_threads(4)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data/data.txt"
CHECKPOINT = ROOT_DIR / "models/v1/glyph.pt"

text = (
    open(DATA_FILE, "r", encoding="utf-8", errors="ignore").read().lower()[:5_000_000]
)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
V = len(chars)

BLOCK = 64
N_LAYER = 3
N_HEAD = 4
N_EMB = 128


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMB, 3 * N_EMB, bias=False)
        self.proj = nn.Linear(N_EMB, N_EMB, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMB, dim=2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMB)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(N_EMB)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMB, 4 * N_EMB), nn.GELU(), nn.Linear(4 * N_EMB, N_EMB)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Glyph(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, N_EMB)
        self.pos = nn.Embedding(BLOCK, N_EMB)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMB)
        self.head = nn.Linear(N_EMB, V, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.tok(idx) + self.pos(pos)
        x = self.ln_f(self.blocks(x))
        return self.head(x)


model = Glyph()
ck = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(ck["model"])
model.eval()
print(f"Loaded Glyph @ step {ck['step']}")


@torch.no_grad()
def generate(prompt, n=500, temp=0.8, top_k=40):
    ids = [stoi.get(c, 0) for c in prompt.lower()]
    out = list(prompt)
    for _ in range(n):
        ctx = ids[-BLOCK:]
        x = torch.tensor([ctx], dtype=torch.long)
        logits = model(x)[0, -1] / temp
        v, idx = torch.topk(logits, top_k)
        probs = F.softmax(v, dim=-1)
        pick = torch.multinomial(probs, 1).item()
        nxt = idx[pick].item()
        ids.append(nxt)
        out.append(itos[nxt])
    return "".join(out)


print("Glyph ready. 'quit' to exit.")
while True:
    s = input("> ").strip()
    if s == "quit":
        break
    if not s:
        continue
    print(generate(s, n=500, temp=0.8, top_k=40))
    print()
