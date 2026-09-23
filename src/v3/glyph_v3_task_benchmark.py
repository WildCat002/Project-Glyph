from __future__ import annotations
import argparse, re, statistics
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
V3A = ROOT / "models/v3/glyph_v3_lstm_best.pt"
V3B = ROOT / "models/v3/glyph_v3b_best.pt"
V3BL = ROOT / "models/v3/glyph_v3b_last.pt"
OUT = ROOT / "results/v3/v3_task_benchmark.txt"
SEEDS = [42,43,44,45,46]
TEMP, TOPK, MAX_NEW = 0.7, 20, 100

TASKS = [
("arith","user: what is 7 + 8?\nassistant:","15"),
("arith","user: what is 80 - 4?\nassistant:","76"),
("arith","user: what is 12 times 4?\nassistant:","48"),
("arith","user: what is 81 divided by 9?\nassistant:","9"),
("yesno","user: is 17 greater than 9?\nassistant:","yes"),
("yesno","user: is 4 less than 2?\nassistant:","no"),
("compare","user: which is larger, 30 or 87?\nassistant:","87"),
("compare","user: which is smaller, 23 or 76?\nassistant:","23"),
("seq","user: what comes next: 2, 5, 8, 11, ?\nassistant:","14"),
("seq","user: what comes next: 10, 16, 22, 28, ?\nassistant:","34"),
("string","user: reverse the word stone.\nassistant:","enots"),
("string","user: how many letters are in apple?\nassistant:","5"),
("logic","user: all zibs are blue. alice is a zib. what color is alice?\nassistant:","blue"),
("logic","user: alice is taller than bob. bob is taller than carol. who is tallest?\nassistant:","alice"),
("logic","user: bob arrives before carol. carol arrives before david. who arrives first?\nassistant:","bob"),
("chat","user: hello\nassistant:","hello"),
("chat","user: thank you\nassistant:","welcome"),
("chat","user: goodbye\nassistant:","goodbye"),
]

class M(nn.Module):
    def __init__(self,v,e,h,l,d):
        super().__init__()
        self.tok=nn.Embedding(v,e)
        self.rnn=nn.LSTM(e,h,l,batch_first=True,dropout=d if l>1 else 0)
        self.head=nn.Linear(h,v)
    def forward(self,x,h=None):
        x,h=self.rnn(self.tok(x),h); return self.head(x),h

def load(path):
    ck=torch.load(path,map_location="cpu",weights_only=False); c=ck["config"]
    chars=c["chars"]; stoi={x:i for i,x in enumerate(chars)}; itos={i:x for x,i in stoi.items()}
    m=M(len(chars),int(c["emb"]),int(c["hidden"]),int(c["layers"]),float(c["dropout"]))
    m.load_state_dict(ck["model"]); m.eval()
    return m,stoi,itos,int(c["seq_len"]),ck

@torch.inference_mode()
def gen(m,stoi,itos,seq,prompt,seed):
    torch.manual_seed(seed)
    ids=[stoi[x] for x in prompt.lower() if x in stoi][-seq:]
    x=torch.tensor([ids],dtype=torch.long)
    z,h=m(x); out=[]
    for _ in range(MAX_NEW):
        z1=z[0,-1].float()/TEMP
        v,i=torch.topk(z1,min(TOPK,z1.numel()))
        nxt=i[torch.multinomial(F.softmax(v,dim=-1),1)].item()
        out.append(itos[nxt])
        z,h=m(torch.tensor([[nxt]]),h)
    return "".join(out)

def first_answer(s):
    s=re.split(r"\n\s*user\s*:",s,maxsplit=1,flags=re.I)[0]
    for line in s.splitlines():
        line=line.strip()
        if line:return line
    return ""

def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^\w\s'-]","",s.lower())).strip()

def ok(cat,exp,ans):
    a=norm(ans); e=norm(exp)
    if cat in ("arith","compare"):
        m=re.search(r"(?<!\d)-?\d+(?!\d)",a)
        return bool(m and int(m.group())==int(e))
    if cat=="yesno": return a.split()[0]==e if a.split() else False
    if cat=="string":
        if e.isdigit():
            m=re.search(r"\d+",a); return bool(m and m.group()==e)
        return a.startswith(e)
    if cat in ("logic","chat"): return e in a[:80]
    if cat=="seq":
        m=re.search(r"-?\d+",a); return bool(m and m.group()==e)
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v3a",type=Path,default=V3A)
    ap.add_argument("--v3b-best",type=Path,default=V3B)
    ap.add_argument("--v3b-last",type=Path,default=V3BL)
    ap.add_argument("--output",type=Path,default=OUT)
    args=ap.parse_args()
    models=[("V3-A best",args.v3a),("V3-B best",args.v3b_best),("V3-B final",args.v3b_last)]
    report=["Glyph V3 EXACT TASK BENCHMARK","="*72,f"temperature={TEMP}",f"top_k={TOPK}",f"seeds={SEEDS}",""]
    summary=[]
    for name,path in models:
        if not path.exists():
            report += [f"SKIP {name}: {path}",""]; continue
        m,stoi,itos,seq,ck=load(path)
        report += [f"[{name}]","step="+str(ck.get("step","?")),f"best_val_loss={ck.get('best_val_loss','?')}",""]
        hits=[]; cats={}
        for ti,(cat,prompt,exp) in enumerate(TASKS):
            per=[]
            for seed in SEEDS:
                out=gen(m,stoi,itos,seq,prompt,seed+ti*1000); ans=first_answer(out); hit=ok(cat,exp,ans)
                per.append(hit); hits.append(hit)
                report += [f"{cat} seed={seed} {'PASS' if hit else 'FAIL'}",f"prompt={prompt!r}",f"expected={exp!r}",f"answer={ans!r}",""]
            cats.setdefault(cat,[]).extend(per)
        score=statistics.mean(hits); summary.append((name,score))
        report += ["- "*36,f"{name} overall: {sum(hits)}/{len(hits)} = {score:.4f}",""]
        for cat,vals in cats.items():
            report.append(f"{name} {cat}: {sum(vals)}/{len(vals)} = {statistics.mean(vals):.4f}")
        report.append("")
    report += ["SUMMARY","="*72]+[f"{n:16s}: {s:.4f}" for n,s in summary]
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text("\n".join(report),encoding="utf-8")
    print(f"Saved: {args.output.resolve()}")

if __name__=="__main__": main()
