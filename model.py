"""Model format, NumPy reference forward pass, and hand-written backpropagation."""
import struct
from pathlib import Path
import numpy as np

NAMES=('embedding','position','q','k','v','out','ff1','ff2','head')

def shapes(d=32,h=4,f=64,context=128,vocab=256):
    return [(vocab,d),(context,d),(d,d),(d,d),(d,d),(d,d),(f,d),(d,f),(vocab,d)]

def initialize(seed=7):
    rng=np.random.default_rng(seed)
    return {name:rng.normal(0,.08,shape).astype(np.float32) for name,shape in zip(NAMES,shapes())}

def save(path,weights,quantized=False):
    with open(path,'wb') as out:
        out.write(b'PTLM0001'+struct.pack('<6I',32,4,64,128,256,int(quantized)))
        for name in NAMES:
            w=weights[name]
            if quantized:
                scale=np.maximum(np.max(np.abs(w),axis=1)/127,1e-10).astype('<f4')
                out.write(scale.tobytes());out.write(np.clip(np.rint(w/scale[:,None]),-127,127).astype('i1').tobytes())
            else:out.write(w.astype('<f4').tobytes())

def load(path):
    with open(path,'rb') as source:
        if source.read(8)!=b'PTLM0001':raise ValueError('Not a Pocket Transformer model')
        dims=struct.unpack('<6I',source.read(24))
        if dims[:5]!=(32,4,64,128,256) or dims[5] not in (0,1):raise ValueError('Unsupported model')
        weights={}
        for name,(rows,cols) in zip(NAMES,shapes()):
            if dims[5]:
                scale=np.frombuffer(source.read(rows*4),dtype='<f4')
                matrix=np.frombuffer(source.read(rows*cols),dtype='i1').reshape(rows,cols)*scale[:,None]
            else:matrix=np.frombuffer(source.read(rows*cols*4),dtype='<f4').reshape(rows,cols).copy()
            weights[name]=matrix
        if source.read(1):raise ValueError('Unexpected trailing data')
    return weights

def rms(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)
def rms_backward(x,g):
    inv=1/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)
    return g*inv-x*(inv**3)*np.mean(g*x,axis=-1,keepdims=True)

def forward(w,tokens,targets=None):
    b,t=tokens.shape;d=32;heads=4;hd=d//heads
    x=w['embedding'][tokens]+w['position'][np.arange(t)]
    n=rms(x)
    def split(a):return a.reshape(b,t,heads,hd).transpose(0,2,1,3)
    def join(a):return a.transpose(0,2,1,3).reshape(b,t,d)
    q,k,v=(split(n@w[name].T) for name in ('q','k','v'))
    scores=q@k.transpose(0,1,3,2)/np.sqrt(hd)
    mask=np.triu(np.ones((t,t),dtype=bool),1)
    scores=np.where(mask,-1e9,scores);scores-=scores.max(axis=-1,keepdims=True)
    attention=np.exp(scores);attention/=attention.sum(axis=-1,keepdims=True)
    merged=join(attention@v);residual=x+merged@w['out'].T
    normalized=rms(residual);pre=normalized@w['ff1'].T;hidden=np.maximum(pre,0)
    final=residual+hidden@w['ff2'].T;last=rms(final);logits=last@w['head'].T
    if targets is None:return logits
    # Cross entropy, then reverse-mode derivatives of each operation above.
    p=np.exp(logits-logits.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
    flat=p.reshape(-1,256);labels=targets.reshape(-1)
    loss=-np.log(flat[np.arange(b*t),labels]+1e-12).mean()
    flat[np.arange(b*t),labels]-=1;g=flat.reshape(b,t,256)/(b*t)
    grads={}
    def linear_backward(name,inputs,grad):
        grads[name]=grad.reshape(-1,grad.shape[-1]).T@inputs.reshape(-1,inputs.shape[-1])
        return grad@w[name]
    gf=rms_backward(final,linear_backward('head',last,g))
    gh=linear_backward('ff2',hidden,gf);gp=gh*(pre>0)
    gr=gf+rms_backward(residual,linear_backward('ff1',normalized,gp))
    gm=linear_backward('out',merged,gr);ga=split(gm)
    dattn=ga@v.transpose(0,1,3,2);dv=attention.transpose(0,1,3,2)@ga
    ds=attention*(dattn-(dattn*attention).sum(axis=-1,keepdims=True));ds=np.where(mask,0,ds)/np.sqrt(hd)
    dq=ds@k;dk=ds.transpose(0,1,3,2)@q
    gn=sum(linear_backward(name,n,join(grad)) for name,grad in zip(('q','k','v'),(dq,dk,dv)))
    gx=gr+rms_backward(x,gn)
    grads['embedding']=np.zeros_like(w['embedding']);np.add.at(grads['embedding'],tokens,gx)
    grads['position']=np.zeros_like(w['position']);grads['position'][:t]=gx.sum(axis=0)
    return float(loss),grads
