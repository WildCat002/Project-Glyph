from __future__ import annotations
import json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BOOK=ROOT/"data/data.txt"; OUT=ROOT/"data/v3"
SEED=42; TRAIN_CHARS=2_500_000; VAL_CHARS=150_000
NAMES=["alice","bob","carol","david","emma","frank","grace","henry","irene","jack","kate","leo"]

def rec(u,a): return f"user: {u}\nassistant: {a}\n\n"

def arithmetic(r):
    k=r.randrange(8)
    if k==0:
        a,b=r.randint(0,999),r.randint(0,999); return f"what is {a} + {b}?",f"{a+b}."
    if k==1:
        a,b=r.randint(0,999),r.randint(0,999); a,b=max(a,b),min(a,b); return f"what is {a} - {b}?",f"{a-b}."
    if k==2:
        a,b=r.randint(0,99),r.randint(0,12); return f"what is {a} times {b}?",f"{a*b}."
    if k==3:
        b,q=r.randint(1,12),r.randint(1,99); return f"what is {b*q} divided by {b}?",f"{q}."
    if k==4:
        a,b=r.randint(0,999),r.randint(0,999); return f"is {a} greater than {b}?","yes." if a>b else "no."
    if k==5:
        a,b=r.randint(0,999),r.randint(0,999); return f"is {a} less than {b}?","yes." if a<b else "no."
    if k==6:
        a,b=r.randint(0,999),r.randint(0,999); return f"what is the difference between {a} and {b}?",f"{abs(a-b)}."
    a,b=r.randint(0,99),r.randint(0,99); return f"what is {a} plus {b}?",f"{a+b}."

def comparison(r):
    a,b=r.sample(range(1,1000),2)
    return (f"which is larger, {a} or {b}?",f"{max(a,b)} is larger.") if r.random()<.5 else (f"which is smaller, {a} or {b}?",f"{min(a,b)} is smaller.")

def sequence(r):
    start,step=r.randint(0,50),r.randint(1,20); xs=[start+step*i for i in range(4)]
    return f"what comes next: {', '.join(map(str,xs))}, ?",f"{xs[-1]+step}."

def logic(r):
    a,b,c=r.sample(NAMES,3); k=r.randrange(5)
    if k==0:return f"{a} is taller than {b}. {b} is taller than {c}. who is tallest?",f"{a} is tallest."
    if k==1:return f"{a} arrives before {b}. {b} arrives before {c}. who arrives first?",f"{a} arrives first."
    if k==2:return f"{a} has 3 apples and gets 2 more. how many apples does {a} have?","5 apples."
    if k==3:return "the box is on the table. where is the box?","the box is on the table."
    return f"all zibs are blue. {a} is a zib. what color is {a}?","blue."

def strings(r):
    w=r.choice(["cat","book","tree","blue","hello","world","apple","train","stone","garden","brain","water","house","river"])
    k=r.randrange(3)
    if k==0:return f"reverse the word {w}.",f"{w[::-1]}."
    if k==1:return f"how many letters are in {w}?",f"{len(w)} letters."
    return f"spell the word {w} one letter at a time."," ".join(w)+"."

def conversation(r):
    items=[
      ("hello",["hello! how can i help you?","hello! what would you like to know?","hi! how can i help?"]),
      ("hi",["hello! how can i help?","hi! what can i do for you?"]),
      ("thank you",["you are welcome.","glad to help.","happy to help."]),
      ("goodbye",["goodbye!","see you later.","goodbye! have a good day."]),
      ("what can you do?",["i can answer questions, follow simple instructions, and help with small tasks.","i can answer questions and help with simple instructions."]),
      ("tell me something interesting.",["octopuses have three hearts.","a day on venus is longer than its year.","rain helps water plants and the ground."]),
      ("say something about rain.",["rain falls from clouds and helps water the ground.","rain can make roads wet and help plants grow."]),
    ]
    u,a=r.choice(items); return u,r.choice(a)

def instruction(r):
    items=[
      ("write one short sentence about a cat.",["the cat sat quietly.","the cat walked across the room."]),
      ("write one short sentence about a tree.",["the tree had green leaves.","the tree moved in the wind."]),
      ("answer with one word: yes or no. is water wet?",["yes."]),
      ("answer with one word: yes or no. can a stone swim?",["no."]),
      ("use the word blue in one sentence.",["the sky looked blue.","the blue book was on the table."]),
    ]
    u,a=r.choice(items); return u,r.choice(a)

GEN=[(arithmetic,28),(comparison,12),(sequence,10),(logic,15),(strings,10),(conversation,15),(instruction,10)]

def build(n,seed):
    r=random.Random(seed); parts=[]; total=0
    funcs=[x[0] for x in GEN]; weights=[x[1] for x in GEN]
    while total<n:
        u,a=r.choices(funcs,weights=weights,k=1)[0](r)
        s=rec(u,a); parts.append(s); total+=len(s)
    r.shuffle(parts); return "".join(parts)[:n]

def main():
    if not BOOK.exists(): raise FileNotFoundError(BOOK)
    OUT.mkdir(parents=True,exist_ok=True)
    tr=build(TRAIN_CHARS,SEED); va=build(VAL_CHARS,SEED+1)
    (OUT/"v3c_dialogue_train.txt").write_text(tr,encoding="utf-8")
    (OUT/"v3c_dialogue_val.txt").write_text(va,encoding="utf-8")
    (OUT/"v3c_manifest.json").write_text(json.dumps({"seed":SEED,"train_chars":len(tr),"val_chars":len(va),"format":"user: ...\\nassistant: ...\\n\\n"},indent=2),encoding="utf-8")
    print(f"train chars: {len(tr):,}"); print(f"val chars: {len(va):,}")

if __name__=="__main__": main()
