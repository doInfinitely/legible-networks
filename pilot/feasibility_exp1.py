import sys, glob, os
import numpy as np
from math import comb
root=sys.argv[1]
rng=np.random.default_rng(0)
def fwd(W1,b1,W2,b2,W3,b3,X,t1=1,t2=1,g=1):
    h1=np.tanh(t1*(X@W1+b1)); h2=np.tanh(t2*(h1@W2+b2)); return g*(h2@W3+b3)
def norm(W):  # incoming weights of unit j = column j
    pos=np.clip(W,0,None).sum(0); neg=(-np.clip(W,None,0)).max(0)
    return np.maximum(pos,neg)
def legal(W): return (np.clip(W,0,None).sum(0)<=1+1e-9).all() and (W>=-1-1e-9).all()
def fmt(a,dec): return np.round(a,dec)
for f in sorted(glob.glob(os.path.join(root,'exp1_which_game/networks/*.npz'))):
    z=np.load(f,allow_pickle=False); g=lambda k:z['actor/'+k].astype(np.float64)
    W1,b1,W2,b2,W3,b3=g('dense1/kernel'),g('dense1/bias'),g('dense2/kernel'),g('dense2/bias'),g('logits/kernel'),g('logits/bias')
    # sparse binary observations; synthetic, feasibility only (not real game states)
    X=(rng.random((4000,500))<0.05).astype(np.float64)
    L=fwd(W1,b1,W2,b2,W3,b3,X)
    t1,t2,tg=norm(W1),norm(W2),norm(W3).max()
    A1,B1,A2,B2,A3,B3=W1/t1,b1/t1,W2/t2,b2/t2,W3/tg,b3/tg
    Lt=fwd(A1,B1,A2,B2,A3,B3,X,t1,t2,tg)
    print(os.path.basename(f)[:8],'legal',legal(A1),legal(A2),legal(A3),'maxlogitdiff %.1e'%np.abs(L-Lt).max(),
      'temps W1 %.1f-%.1f W2 %.1f-%.1f gain %.2f'%(t1.min(),t1.max(),t2.min(),t2.max(),tg))
    print('   median |w| raw W1 %.4f  transplant W1 %.5f; frac rounding to 0 @3dp raw %.3f transplant %.3f'%(np.median(np.abs(W1)),np.median(np.abs(A1)),(np.round(W1,3)==0).mean(),(np.round(A1,3)==0).mean()))
    for dec in (3,4,5):
        R=fwd(*[fmt(a,dec) for a in (W1,b1,W2,b2,W3,b3)],X)
        T=fwd(*[fmt(a,dec) for a in (A1,B1,A2,B2,A3,B3)],X,np.round(t1,dec),np.round(t2,dec),round(tg,dec))
        print('   %ddp argmax agreement raw %.4f transplant %.4f'%(dec,(R.argmax(1)==L.argmax(1)).mean(),(T.argmax(1)==L.argmax(1)).mean()))
print('--- binomial, one-sided against chance')
def tail(n,k,p): return sum(comb(n,i)*p**i*(1-p)**(n-i) for i in range(k,n+1))
for n in (3,6,12,20,30):
    k=next(k for k in range(n+2) if tail(n,k,.5)<=.05) if tail(n,n,.5)<=.05 else None
    print(n,'threshold',k, None if k is None else 'alpha %.3f power@.7 %.2f @.8 %.2f @.9 %.2f'%(tail(n,k,.5),tail(n,k,.7),tail(n,k,.8),tail(n,k,.9)))
