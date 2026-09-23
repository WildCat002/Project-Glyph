from __future__ import annotations
import argparse,csv,math,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
BOOK=ROOT/"data/data.txt"; DTR=ROOT/"data/v3/v3c_dialogue_train.txt"; DVA=ROOT/"data/v3/v3c_dialogue_val.txt"
BASE=ROOT/"models/v3/glyph_v3b_best.pt"; BEST=ROOT/"models/v3/glyph_v3c_best.pt"; LAST=ROOT/"models/v3/glyph_v3c_last.pt"; HIST=ROOT/"experiments/v3c/glyph_v3c_history.csv"
SEED=42; SEQ=128; BATCH=32; STEPS=50000; WARMUP=1000; LR=5e-5; MIN_LR=5e-6; WD=.01; CLIP=1.; MIX=.75; VAL_BATCHES=64; LOG=500; SAVE=500

class M(nn.Module):
    def __init__(self,v,e=96,h=240,l=2,d=.1):
        super().__init__(); self.tok=nn.Embedding(v,e); self.rnn=nn.LSTM(e,h,l,batch_first=True,dropout=d if l>1 else 0.); self.head=nn.Linear(h,v)
    def forward(self,x,hid=None): x,hid=self.rnn(self.tok(x),hid); return self.head(x),hid

def batch(data,rng,dev):
    s=torch.randint(0,len(data)-SEQ-1,(BATCH,),generator=rng).tolist()
    x=torch.stack([data[i:i+SEQ] for i in s]); y=torch.stack([data[i+1:i+SEQ+1] for i in s])
    return x.to(dev),y.to(dev)

@torch.no_grad()
def ev(m,data,dev,seed):
    m.eval(); r=torch.Generator(device="cpu"); r.manual_seed(seed); ls=[]
    for _ in range(VAL_BATCHES):
        x,y=batch(data,r,dev); z,_=m(x); ls.append(float(F.cross_entropy(z.reshape(-1,z.size(-1)).float(),y.reshape(-1)).item()))
    m.train(); return sum(ls)/len(ls)

def lr_at(step,total):
    if step<=WARMUP:return LR*step/WARMUP
    p=(step-WARMUP)/max(1,total-WARMUP); return MIN_LR+(LR-MIN_LR)*.5*(1+math.cos(math.pi*p))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--base",type=Path,default=BASE); p.add_argument("--cpu",action="store_true"); p.add_argument("--steps",type=int,default=STEPS); p.add_argument("--dialogue-prob",type=float,default=MIX); a=p.parse_args()
    random.seed(SEED); torch.manual_seed(SEED)
    dev=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    book=BOOK.read_text(encoding="utf-8",errors="ignore").lower(); bt=book[:4750000]; bv=book[-250000:]
    dt=DTR.read_text(encoding="utf-8",errors="ignore").lower(); dv=DVA.read_text(encoding="utf-8",errors="ignore").lower()
    chars=sorted(set(bt+bv)); extra=sorted(set(dt+dv)-set(chars))
    if extra: raise RuntimeError(f"dialogue contains chars outside vocabulary: {extra}")
    stoi={c:i for i,c in enumerate(chars)}; enc=lambda s:torch.tensor([stoi[c] for c in s],dtype=torch.long)
    bt,bv,dt,dv=enc(bt),enc(bv),enc(dt),enc(dv)
    ck=torch.load(a.base,map_location=dev,weights_only=False); c=ck["config"]
    m=M(len(chars),int(c["emb"]),int(c["hidden"]),int(c["layers"]),float(c["dropout"])).to(dev); m.load_state_dict(ck["model"])
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    BEST.parent.mkdir(parents=True,exist_ok=True); HIST.parent.mkdir(parents=True,exist_ok=True)
    HIST.write_text("step,train_loss,book_val_loss,dialogue_val_loss,combined_val_loss,lr,grad_norm\n",encoding="utf-8")
    rng=torch.Generator(device="cpu"); rng.manual_seed(SEED); best=float("inf"); bestd=float("inf"); t0=time.time()
    print("="*68); print("Glyph V3-C Training"); print("="*68)
    print(f"Base checkpoint : {a.base}"); print(f"Device          : {dev}"); print(f"Parameters      : {sum(p.numel() for p in m.parameters()):,}"); print(f"Dialogue mix    : {a.dialogue_prob:.0%}"); print(f"Steps           : {a.steps:,}"); print("="*68)
    state={}
    for step in range(1,a.steps+1):
        src=dt if random.random()<a.dialogue_prob else bt; x,y=batch(src,rng,dev); lr=lr_at(step,a.steps)
        for g in opt.param_groups:g["lr"]=lr
        z,_=m(x); loss=F.cross_entropy(z.reshape(-1,z.size(-1)).float(),y.reshape(-1))
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at {step}")
        opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(m.parameters(),CLIP); opt.step()
        if step%LOG==0 or step==1:
            bv_loss=ev(m,bv,dev,SEED+step); dv_loss=ev(m,dv,dev,SEED+100000+step); combined=(1-a.dialogue_prob)*bv_loss+a.dialogue_prob*dv_loss
            with HIST.open("a",encoding="utf-8",newline="") as f: csv.writer(f).writerow([step,float(loss.item()),bv_loss,dv_loss,combined,lr,float(gn)])
            print(f"step {step:6d} | train {loss.item():.4f} | book_val {bv_loss:.4f} | dialogue_val {dv_loss:.4f} | combined {combined:.4f} | lr {lr:.3e} | grad {float(gn):.3f}")
            state={"model":m.state_dict(),"opt":opt.state_dict(),"step":step,"best_val_loss":best,"best_dialogue_val_loss":bestd,"config":{"seq_len":SEQ,"batch":BATCH,"max_steps":a.steps,"warmup":WARMUP,"lr":LR,"min_lr":MIN_LR,"weight_decay":WD,"grad_clip":CLIP,"dialogue_prob":a.dialogue_prob,"vocab":len(chars),"chars":chars,"emb":96,"hidden":240,"layers":2,"dropout":.1,"base_checkpoint":str(a.base)}}
            if combined<best:
                best=combined; state["best_val_loss"]=best; torch.save(state,BEST); print(f"  saved BEST -> {BEST}")
            if dv_loss<bestd: bestd=dv_loss
        if step%SAVE==0 and state:
            state["best_val_loss"]=best; state["best_dialogue_val_loss"]=bestd; torch.save(state,LAST)
    print("="*68); print("Training complete"); print(f"Best combined validation loss: {best:.6f}"); print(f"Best checkpoint: {BEST}"); print(f"Final checkpoint: {LAST}"); print(f"Elapsed: {(time.time()-t0)/60:.1f} min"); print("="*68)

if __name__=="__main__": main()
