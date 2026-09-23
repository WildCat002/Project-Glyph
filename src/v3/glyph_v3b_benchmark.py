from __future__ import annotations
import argparse,json,re,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

ROOT=Path(__file__).resolve().parents[2];DEFAULT=ROOT/"models/v3/glyph_v3b_best.pt";EVAL=ROOT/"data/v3/v3b_eval.jsonl";OUT=ROOT/"results/v3/benchmark_v3B_results.txt"
class GlyphV3LSTM(nn.Module):
    def __init__(self,vocab,emb,hidden,layers,dropout):
        super().__init__();self.tok=nn.Embedding(vocab,emb);self.rnn=nn.LSTM(emb,hidden,layers,batch_first=True,dropout=dropout if layers>1 else 0.);self.head=nn.Linear(hidden,vocab)
    def forward(self,idx,hidden=None):x,hidden=self.rnn(self.tok(idx),hidden);return self.head(x),hidden
def sample(logits):
    logits=logits.float()/.7;v,i=torch.topk(logits,min(20,logits.numel()));f=torch.full_like(logits,float("-inf"));f[i]=v;return torch.multinomial(F.softmax(f,dim=-1),1).item()
@torch.inference_mode()
def gen(model,prompt,stoi,itos,dev,seq,n=120):
    ids=[stoi[c] for c in prompt.lower() if c in stoi][-seq:];ids=ids or [stoi[" "]];idx=torch.tensor([ids],dtype=torch.long,device=dev);logits,h=model(idx);out=[]
    for _ in range(n):
        nxt=sample(logits[0,-1]);out.append(itos[nxt]);cur=torch.tensor([[nxt]],dtype=torch.long,device=dev);logits,h=model(cur,h)
    return "".join(out)
def hit(exp,out):
    e=exp.lower();c=out.lower()[:120]
    return (re.search(rf"\b{re.escape(e)}\b",c) is not None) if e.isdigit() else e in c
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,default=DEFAULT);p.add_argument("--cpu",action="store_true");a=p.parse_args()
    dev=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda");ck=torch.load(a.checkpoint,map_location=dev,weights_only=False);cfg=ck["config"];chars=cfg["chars"];stoi={c:i for i,c in enumerate(chars)};itos={i:c for i,c in enumerate(chars)}
    m=GlyphV3LSTM(len(chars),int(cfg["emb"]),int(cfg["hidden"]),int(cfg["layers"]),float(cfg["dropout"])).to(dev);m.load_state_dict(ck["model"]);m.eval()
    items=[json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines() if x.strip()];report=["="*68,"Glyph V3-B Benchmark","="*68,f"Checkpoint   : {a.checkpoint}",f"Step         : {ck.get('step','?')}",f"Best val loss: {ck.get('best_val_loss',float('nan')):.6f}",f"Parameters   : {sum(p.numel() for p in m.parameters()):,}",f"Device       : {dev}","="*68];hits=[]
    for item in items:
        out=gen(m,item["prompt"],stoi,itos,dev,int(cfg["seq_len"]));ok=hit(item["expected"],out);hits.append(ok);report += ["",f"PROMPT: {item['prompt']!r}",f"EXPECTED: {item['expected']!r}",f"OUTPUT: {out}",f"HIT: {ok}"]
    acc=statistics.mean(hits);report += ["",f"Expected-answer hit rate: {acc:.4f}",f"Hits: {sum(hits)}/{len(hits)}","", "="*64,"End of benchmark"];OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text("\n".join(report),encoding="utf-8");print("\n"+"="*64);print(f"Benchmark complete. Results saved to: {OUT}");print(f"Expected-answer hit rate: {acc:.4f}");print("="*64)
if __name__=="__main__":main()
