from __future__ import annotations
import argparse,re
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; DEFAULT=ROOT/"models/v3/glyph_v3c_best.pt"

class M(nn.Module):
    def __init__(self,v,e,h,l,d):
        super().__init__(); self.tok=nn.Embedding(v,e); self.rnn=nn.LSTM(e,h,l,batch_first=True,dropout=d if l>1 else 0.); self.head=nn.Linear(h,v)
    def forward(self,x,hid=None): x,hid=self.rnn(self.tok(x),hid); return self.head(x),hid

def nxt(z,t,k):
    z=z.float()/max(t,1e-6); v,i=torch.topk(z,min(k,z.numel())); return i[torch.multinomial(F.softmax(v,dim=-1),1)].item()

@torch.inference_mode()
def gen(m,stoi,itos,seq,prompt,t,k,n):
    ids=[stoi[c] for c in prompt.lower() if c in stoi][-seq:]; x=torch.tensor([ids],dtype=torch.long); z,h=m(x); out=[]
    for _ in range(n):
        j=nxt(z[0,-1],t,k); out.append(itos[j]); text="".join(out)
        if re.search(r"\n\s*user\s*:",text,re.I): break
        z,h=m(torch.tensor([[j]],dtype=torch.long),h)
    return re.split(r"\n\s*user\s*:", "".join(out), maxsplit=1, flags=re.I)[0]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,default=DEFAULT); p.add_argument("--cpu",action="store_true"); p.add_argument("--temperature",type=float,default=.7); p.add_argument("--top-k",type=int,default=20); p.add_argument("--max-new-tokens",type=int,default=200); a=p.parse_args()
    dev=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda"); ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); c=ck["config"]; chars=c["chars"]; stoi={x:i for i,x in enumerate(chars)}; itos={i:x for x,i in stoi.items()}
    m=M(len(chars),int(c["emb"]),int(c["hidden"]),int(c["layers"]),float(c["dropout"])).to(dev); m.load_state_dict(ck["model"]); m.eval()
    print("="*68); print("Glyph V3-C Interactive Chat"); print("="*68); print(f"Checkpoint   : {a.checkpoint}"); print(f"Step         : {ck.get('step','?')}"); print(f"Best val loss: {ck.get('best_val_loss',float('nan')):.6f}"); print(f"Parameters   : {sum(p.numel() for p in m.parameters()):,}"); print(f"Device       : {dev}"); print("="*68)
    t,k,n=a.temperature,a.top_k,a.max_new_tokens
    while True:
        try:s=input("You: ")
        except (EOFError,KeyboardInterrupt): print("\nGoodbye!"); break
        q=s.strip()
        if q==":quit": print("Goodbye!"); break
        if q.startswith(":temp "): t=float(q.split(maxsplit=1)[1]); print(f"Temperature={t}"); continue
        if q.startswith(":topk "): k=int(q.split(maxsplit=1)[1]); print(f"Top-k={k}"); continue
        if q.startswith(":tokens "): n=int(q.split(maxsplit=1)[1]); print(f"New chars={n}"); continue
        print("Glyph:",gen(m,stoi,itos,int(c["seq_len"]),f"user: {s}\nassistant:",t,k,n).strip(),"\n")

if __name__=="__main__": main()
