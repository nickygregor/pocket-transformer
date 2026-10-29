"""Train a tiny byte-level transformer on your own text, using only NumPy."""
import argparse
import json
import numpy as np
from model import initialize,forward,save
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',default=str(Path(__file__).with_name('corpus.txt')))
    p.add_argument('--steps',type=int,default=800);p.add_argument('--output',default='model.bin')
    p.add_argument('--seed',type=int,default=7);args=p.parse_args()
    raw=Path(args.corpus).read_bytes();seq=128;batch=4
    if len(raw)<seq+2 or args.steps<1:p.error('Need at least 130 corpus bytes and positive training steps')
    corpus=np.frombuffer(raw,dtype=np.uint8).astype(np.int64);rng=np.random.default_rng(args.seed)
    w=initialize(args.seed);m={k:np.zeros_like(v) for k,v in w.items()};v={k:np.zeros_like(a) for k,a in w.items()}
    first=None;loss=None
    for step in range(1,args.steps+1):
        starts=rng.integers(0,len(corpus)-seq-1,size=batch);positions=starts[:,None]+np.arange(seq+1)
        chunk=corpus[positions];loss,grad=forward(w,chunk[:,:-1],chunk[:,1:]);first=loss if first is None else first
        total_norm=np.sqrt(sum(np.sum(g*g) for g in grad.values()));clip=min(1,1/(total_norm+1e-8))
        for name in w:
            g=grad[name]*clip;m[name]=.9*m[name]+.1*g;v[name]=.999*v[name]+.001*g*g
            w[name]-=.003*(m[name]/(1-.9**step))/(np.sqrt(v[name]/(1-.999**step))+1e-8)
        if step==1 or step%100==0:print(f'step {step:4} · training loss {loss:.4f}',flush=True)
    save(args.output,w)
    metadata={'seed':args.seed,'steps':args.steps,'corpus_bytes':len(raw),'first_training_loss':first,'last_training_loss':loss,
              'note':'Training-set loss only; tiny educational model, not a general-purpose LLM.'}
    Path(args.output+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved {args.output}')
if __name__=='__main__':main()
