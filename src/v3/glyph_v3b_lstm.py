from __future__ import annotations
import argparse,csv,math,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
BOOK=ROOT/"data/data.txt"; DTRAIN=ROOT/"data/v3/v3b_conversations_train.txt"; DVAL=ROOT/"data/v3/v3b_conversations_val.txt"
BASE=ROOT/"models/v3/glyph_v3_lstm_best.pt"; BEST=ROOT/"models/v3/glyph_v3b_best.pt"; LAST=ROOT/"models/v3/glyph_v3b_last.pt"; HIST=ROOT/"experiments/v3b/glyph_v3b_history.csv"
SEED=42; SEQ=128; BATCH=32; STEPS=50000; WARMUP=1000; LR=1e-4; MIN_LR=1e-5; WD=.01; CLIP=1.; MIX=.25; VAL_BATCHES=64; LOG=500; SAVE=500

class GlyphV3LSTM(nn.Module):
    def __init__(self,vocab,emb=96,hidden=240,layers=2,dropout=.1):
        super().__init__(); self.tok=nn.Embedding(vocab,emb); self.rnn=nn.LSTM(emb,hidden,layers,batch_first=True,dropout=dropout if layers>1 else 0.); self.head=nn.Linear(hidden,vocab)
    def forward(self,idx,hidden=None):
        x,hidden=self.rnn(self.tok(idx),hidden); return self.head(x),hidden

def batch(data,n,seq,dev,rng):
    starts=torch.randint(0,len(data)-seq-1,(n,),generator=rng).tolist()
    x=torch.stack([data[s:s+seq] for s in starts]); y=torch.stack([data[s+1:s+seq+1] for s in starts])
    return x.to(dev),y.to(dev)

@torch.no_grad()
def evaluate(model,data,batches,dev,seed):
    model.eval(); rng=torch.Generator(device="cpu"); rng.manual_seed(seed); ls=[]
    for _ in range(batches):
        x,y=batch(data,BATCH,SEQ,dev,rng); z,_=model(x)
        ls.append(float(F.cross_entropy(z.reshape(-1,z.size(-1)).float(),y.reshape(-1)).item()))
    model.train(); return sum(ls)/len(ls)

def lr_at(step,total):
    if step<=WARMUP:return LR*step/WARMUP
    p=(step-WARMUP)/max(1,total-WARMUP); return MIN_LR+(LR-MIN_LR)*.5*(1+math.cos(math.pi*p))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--base",type=Path,default=BASE); p.add_argument("--cpu",action="store_true"); p.add_argument("--steps",type=int,default=STEPS); p.add_argument("--dialogue-prob",type=float,default=MIX); a=p.parse_args()
    random.seed(SEED); torch.manual_seed(SEED); dev=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    book=BOOK.read_text(encoding="utf-8",errors="ignore").lower(); bt=book[:4750000]; bv=book[-250000:]
    dt=DTRAIN.read_text(encoding="utf-8",errors="ignore").lower(); dv=DVAL.read_text(encoding="utf-8",errors="ignore").lower()
    chars=sorted(set(bt+bv)); extra=sorted((set(dt+dv)-set(chars)))
    if extra: raise RuntimeError(f"dialogue chars not in V3 vocab: {extra}")
    stoi={c:i for i,c in enumerate(chars)}
    enc=lambda s:torch.tensor([stoi[c] for c in s],dtype=torch.long)
    bt,bv,dt,dv=map(enc,(bt,bv,dt,dv))
    ck=torch.load(a.base,map_location=dev,weights_only=False); cfg=ck["config"]
    model=GlyphV3LSTM(len(chars),int(cfg["emb"]),int(cfg["hidden"]),int(cfg["layers"]),float(cfg["dropout"])).to(dev); model.load_state_dict(ck["model"])
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    BEST.parent.mkdir(parents=True,exist_ok=True); HIST.parent.mkdir(parents=True,exist_ok=True); HIST.write_text("step,train_loss,book_val_loss,dialogue_val_loss,combined_val_loss,lr,grad_norm\n",encoding="utf-8")
    rng=torch.Generator(device="cpu"); rng.manual_seed(SEED); best=float("inf"); best_d=float("inf"); started=time.time()
    print("="*68); print("Glyph V3-B Training"); print("="*68); print(f"Base checkpoint : {a.base}"); print(f"Device          : {dev}"); print(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}"); print(f"Vocabulary      : {len(chars)}"); print(f"Dialogue mix    : {a.dialogue_prob:.2%}"); print(f"Max steps       : {a.steps:,}"); print("="*68)
    for step in range(1,a.steps+1):
        src=dt if random.random()<a.dialogue_prob else bt; x,y=batch(src,BATCH,SEQ,dev,rng); lr=lr_at(step,a.steps)
        for g in opt.param_groups:g["lr"]=lr
        z,_=model(x); loss=F.cross_entropy(z.reshape(-1,z.size(-1)).float(),y.reshape(-1))
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP); opt.step()
        if step%LOG==0 or step==1:
            bl=evaluate(model,bv,VAL_BATCHES,dev,SEED+step); dl=evaluate(model,dv,VAL_BATCHES,dev,SEED+100000+step); comb=(1-a.dialogue_prob)*bl+a.dialogue_prob*dl
            with HIST.open("a",encoding="utf-8",newline="") as f:csv.writer(f).writerow([step,float(loss.item()),bl,dl,comb,lr,float(gn)])
            print(f"step {step:6d} | train {loss.item():.4f} | book_val {bl:.4f} | dialogue_val {dl:.4f} | combined {comb:.4f} | lr {lr:.3e} | grad {float(gn):.3f}")
            state={"model":model.state_dict(),"opt":opt.state_dict(),"step":step,"best_val_loss":best,"best_dialogue_val_loss":best_d,"config":{"seq_len":SEQ,"batch":BATCH,"max_steps":a.steps,"warmup":WARMUP,"lr":LR,"min_lr":MIN_LR,"weight_decay":WD,"grad_clip":CLIP,"dialogue_prob":a.dialogue_prob,"vocab":len(chars),"chars":chars,"emb":96,"hidden":240,"layers":2,"dropout":.1,"base_checkpoint":str(a.base)}}
            if comb<best:best=comb;state["best_val_loss"]=best;torch.save(state,BEST);print(f"  saved BEST -> {BEST}")
            if dl<best_d:best_d=dl
        if step%SAVE==0:
            state["best_val_loss"]=best;state["best_dialogue_val_loss"]=best_d;torch.save(state,LAST)
    print("="*68);print("Training complete");print(f"Best combined validation loss: {best:.6f}");print(f"Best checkpoint              : {BEST}");print(f"Last checkpoint              : {LAST}");print(f"Elapsed                      : {(time.time()-started)/60:.1f} min");print("="*68)

if __name__=="__main__":main()
