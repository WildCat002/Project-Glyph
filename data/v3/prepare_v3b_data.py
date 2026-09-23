from __future__ import annotations
import json
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "data/v3"
TRAIN_CHARS = 1_250_000
VAL_CHARS = 75_000
SEED = 42
NAMES = ["alice","bob","carol","david","emma","frank","grace","henry","irene","jack","kate","leo"]

def arithmetic(rng):
    k = rng.randrange(6)
    if k == 0:
        a,b=rng.randint(0,99),rng.randint(0,99); return f"what is {a} + {b}?",f"{a+b}."
    if k == 1:
        a,b=rng.randint(0,99),rng.randint(0,99); a,b=max(a,b),min(a,b); return f"what is {a} - {b}?",f"{a-b}."
    if k == 2:
        a,b=rng.randint(0,20),rng.randint(0,20); return f"what is {a} times {b}?",f"{a*b}."
    if k == 3:
        b,q=rng.randint(1,12),rng.randint(1,12); return f"what is {b*q} divided by {b}?",f"{q}."
    a,b=rng.randint(1,99),rng.randint(1,99)
    if k == 4: return f"is {a} greater than {b}?","yes." if a>b else "no."
    return f"is {a} less than {b}?","yes." if a<b else "no."

def compare(rng):
    a,b=rng.sample(range(1,100),2)
    return (f"which is larger, {a} or {b}?", f"{max(a,b)} is larger.") if rng.random()<.5 else (f"which is smaller, {a} or {b}?", f"{min(a,b)} is smaller.")

def logic(rng):
    a,b,c=rng.sample(NAMES,3); k=rng.randrange(5)
    if k==0: return f"{a} is taller than {b}. {b} is taller than {c}. who is tallest?",f"{a} is tallest."
    if k==1: return f"all zibs are blue. {a} is a zib. what color is {a}?","blue."
    if k==2: return f"{a} has 3 apples and gets 2 more. how many apples does {a} have?","5 apples."
    if k==3: return "the box is on the table. where is the box?","the box is on the table."
    return f"{a} arrives before {b}. {b} arrives before {c}. who arrives first?",f"{a} arrives first."

def sequence(rng):
    start,step=rng.randint(0,20),rng.randint(1,9); xs=[start+step*i for i in range(4)]
    return f"what comes next: {', '.join(map(str,xs))}, ?",f"{xs[-1]+step}."

def string_task(rng):
    word=rng.choice(["cat","book","tree","red","blue","hello","world","apple","train","stone"]); k=rng.randrange(3)
    if k==0:return f"reverse the word {word}.",f"{word[::-1]}."
    if k==1:return f"how many letters are in {word}?",f"{len(word)} letters."
    return f"spell the word {word} one letter at a time."," ".join(word)+"."

def convo(rng):
    items=[
        ("hello",["hello! how can i help you today?","hello! it is nice to talk with you.","hi! what would you like to do?"]),
        ("hi",["hi! how can i help?","hello! what can i do for you?"]),
        ("thank you",["you are welcome.","glad to help."]),
        ("thanks",["you are welcome.","happy to help."]),
        ("goodbye",["goodbye!","see you later.","goodbye! have a good day."]),
        ("what can you do?",["i can answer questions, follow simple instructions, and continue text.","i can help with simple questions and small reasoning tasks."]),
        ("tell me something interesting.",["a day on venus is longer than its year.","octopuses have three hearts.","honey can remain edible for a very long time when stored properly."]),
        ("say something about rain.",["rain falls from clouds and helps water the ground.","rain can make roads wet and help plants grow."]),
    ]
    u,answers=rng.choice(items); return u,rng.choice(answers)

def instruction(rng):
    items=[
        ("write one short sentence about a cat.",["the cat sat quietly.","the cat walked across the room.","the cat watched the rain."]),
        ("write one short sentence about a tree.",["the tree stood beside the road.","the tree had green leaves.","the tree moved in the wind."]),
        ("write one short sentence about the sea.",["the sea was calm in the morning.","the sea moved gently near the shore.","the sea looked wide and blue."]),
        ("answer with one word: yes or no. is water wet?",["yes."]),
        ("answer with one word: yes or no. can a stone swim?",["no."]),
    ]
    u,answers=rng.choice(items); return u,rng.choice(answers)

def build(target,seed):
    rng=random.Random(seed); gens=[arithmetic,arithmetic,compare,logic,logic,sequence,string_task,convo,convo,instruction]
    parts=[]; total=0
    while total<target:
        u,a=rng.choice(gens)(rng); s=f"user: {u}\nassistant: {a}\n\n"; parts.append(s); total+=len(s)
    rng.shuffle(parts); return "".join(parts)[:target]

def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    train=build(TRAIN_CHARS,SEED); val=build(VAL_CHARS,SEED+1)
    (OUT_DIR/"v3b_conversations_train.txt").write_text(train,encoding="utf-8")
    (OUT_DIR/"v3b_conversations_val.txt").write_text(val,encoding="utf-8")
    evals=[
        {"prompt":"user: what is 7 + 8?\nassistant:","expected":"15"},
        {"prompt":"user: what is 12 times 4?\nassistant:","expected":"48"},
        {"prompt":"user: what is 81 divided by 9?\nassistant:","expected":"9"},
        {"prompt":"user: is 17 greater than 9?\nassistant:","expected":"yes."},
        {"prompt":"user: is 4 less than 2?\nassistant:","expected":"no."},
        {"prompt":"user: all zibs are blue. alice is a zib. what color is alice?\nassistant:","expected":"blue."},
        {"prompt":"user: alice is taller than bob. bob is taller than carol. who is tallest?\nassistant:","expected":"alice is tallest."},
        {"prompt":"user: bob arrives before carol. carol arrives before david. who arrives first?\nassistant:","expected":"bob arrives first."},
        {"prompt":"user: what comes next: 2, 5, 8, 11, ?\nassistant:","expected":"14."},
        {"prompt":"user: reverse the word stone.\nassistant:","expected":"enots."},
        {"prompt":"user: how many letters are in apple?\nassistant:","expected":"5 letters."},
        {"prompt":"user: hello\nassistant:","expected":"hello"},
        {"prompt":"user: thank you\nassistant:","expected":"welcome"},
        {"prompt":"user: goodbye\nassistant:","expected":"goodbye"},
        {"prompt":"user: answer with one word: yes or no. is water wet?\nassistant:","expected":"yes."},
        {"prompt":"user: answer with one word: yes or no. can a stone swim?\nassistant:","expected":"no."},
    ]
    (OUT_DIR/"v3b_eval.jsonl").write_text("".join(json.dumps(x)+"\n" for x in evals),encoding="utf-8")
    (OUT_DIR/"v3b_data_manifest.json").write_text(json.dumps({"seed":SEED,"train_chars":len(train),"val_chars":len(val),"eval_items":len(evals)},indent=2),encoding="utf-8")
    print("V3-B data prepared:",len(train),len(val),len(evals))

if __name__=="__main__": main()
