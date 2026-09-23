from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/"models/v3/glyph_v3b_best.pt"

class GlyphV3LSTM(nn.Module):
    def __init__(self,vocab,emb,hidden,layers,dropout):
        super().__init__(); self.tok=nn.Embedding(vocab,emb); self.rnn=nn.LSTM(emb,hidden,layers,batch_first=True,dropout=dropout if layers>1 else 0.); self.head=nn.Linear(hidden,vocab)
    def forward(self,idx,hidden=None):
        x,hidden=self.rnn(self.tok(idx),hidden); return self.head(x),hidden

def sample(logits,temp,topk):
    logits=logits.float()/max(temp,1e-6); k=min(topk,logits.numel()); v,i=torch.topk(logits,k); f=torch.full_like(logits,float("-inf")); f[i]=v
    return torch.multinomial(F.softmax(f,dim=-1),1).item()

@torch.inference_mode()
def generate(model,prompt,stoi,itos,dev,seq,temp,topk,n):
    ids=[stoi[c] for c in prompt.lower() if c in stoi][-seq:]; ids=ids or [stoi[" "]]
    idx=torch.tensor([ids],dtype=torch.long,device=dev); logits,h=model(idx); out=[]
    for _ in range(n):
        nxt=sample(logits[0,-1],temp,topk); out.append(itos[nxt]); cur=torch.tensor([[nxt]],dtype=torch.long,device=dev); logits,h=model(cur,h)
    return "".join(out)

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,default=DEFAULT);p.add_argument("--cpu",action="store_true");p.add_argument("--temperature",type=float,default=.7);p.add_argument("--top-k",type=int,default=20);p.add_argument("--max-new-tokens",type=int,default=300);a=p.parse_args()
    dev=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda");ck=torch.load(a.checkpoint,map_location=dev,weights_only=False);cfg=ck["config"];chars=cfg["chars"];stoi={c:i for i,c in enumerate(chars)};itos={i:c for i,c in enumerate(chars)}
    model=GlyphV3LSTM(len(chars),int(cfg["emb"]),int(cfg["hidden"]),int(cfg["layers"]),float(cfg["dropout"])).to(dev);model.load_state_dict(ck["model"]);model.eval()
    print("="*68);print("Glyph V3-B Interactive Chat");print("="*68);print(f"Checkpoint   : {a.checkpoint}");print(f"Step         : {ck.get('step','?')}");print(f"Best val loss: {ck.get('best_val_loss',float('nan')):.6f}");print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}");print(f"Device       : {dev}");print(f"Temperature  : {a.temperature}");print(f"Top-k        : {a.top_k}");print(f"New chars    : {a.max_new_tokens}");print("="*68)
    temp,topk,n=a.temperature,a.top_k,a.max_new_tokens
    while True:
        try:s=input("You: ")
        except (EOFError,KeyboardInterrupt):print("\nGoodbye!");break
        cmd=s.strip()
        if cmd==":quit":print("Goodbye!");break
        if cmd.startswith(":temp "):temp=float(cmd.split(maxsplit=1)[1]);print(f"Temperature = {temp}");continue
        if cmd.startswith(":topk "):topk=int(cmd.split(maxsplit=1)[1]);print(f"Top-k = {topk}");continue
        if cmd.startswith(":tokens "):n=int(cmd.split(maxsplit=1)[1]);print(f"New chars = {n}");continue
        print("Glyph:",generate(model,f"user: {s}\nassistant:",stoi,itos,dev,int(cfg["seq_len"]),temp,topk,n),"\n")

if __name__=="__main__":main()
