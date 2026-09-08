"""Local polynomial theory acceleration, followed by exact importance correction.

The emulator supplies an MCMC proposal density; final retained samples have
direct CAMB likelihood evaluations and importance weights p_exact/p_emulator.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,itertools,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.linalg import lstsq
from model import ROOT,CENTER,SCALE,FID,logprior,Likelihood

def powers(degree=3):
 out=[np.zeros(8,dtype=int)]
 for d in range(1,degree+1):
  for c in itertools.combinations_with_replacement(range(8),d):
   out.append(np.bincount(c,minlength=8))
 return np.array(out)

def design(t,exponents):
 t=np.atleast_2d(t)
 X=np.ones((len(t),len(exponents)))
 for j in range(8):
  X*=t[:,j,None]**exponents[None,:,j]
 return X

class Emulator:
 def __init__(self,path=None):
  f=np.load(path or ROOT/'results/emulator.npz')
  for k in ['center','scale','powers','coeff','mean']:setattr(self,k,f[k])
  self.indices=[np.repeat(np.arange(8),p) for p in self.powers]
 def __call__(self,x):
  t=(np.asarray(x)-self.center)/self.scale
  return design(t,self.powers)@self.coeff+self.mean
 def one(self,x):return self(x)[0]

L=None
def init_worker():
 global L
 L=Likelihood(with_pantheon=False)

def evaluate(task):
 i,x=task
 try:
  y=L.theory(x)
  np.savez(ROOT/f'results/training/point_{i:04d}.npz',x=x,y=y)
  return i,True
 except Exception as e:
  (ROOT/f'results/training/error_{i:04d}.txt').write_text(repr(e))
  return i,False

def fit():
 files=sorted((ROOT/'results/training').glob('point_*.npz'))
 X=[];Y=[]
 for p in files:
  f=np.load(p);X.append(f['x']);Y.append(f['y'])
 x=np.array(X);y=np.array(Y);po=powers(3)
 D=design((x-CENTER)/SCALE,po);mean=y.mean(axis=0)
 co,_,rank,_=lstsq(D,y-mean,lapack_driver='gelsd')
 np.savez(ROOT/'results/emulator.npz',center=CENTER,scale=SCALE,powers=po,coeff=co,mean=mean)
 np.savez(ROOT/'results/training_bank.npz',x=x,y=y)
 print('fit',len(x),'points',len(po),'terms','rank',rank,'RMS outputs',np.std(D@co+mean-y,axis=0)[-5:],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=640);p.add_argument('--workers',type=int,default=8);p.add_argument('--fit-only',action='store_true');a=p.parse_args()
 if a.fit_only:fit();raise SystemExit
 directory=ROOT/'results/training';directory.mkdir(exist_ok=True)
 rng=np.random.default_rng(20260904)
 points=[FID,CENTER]
 # Latin hypercube is unnecessary for this local, Gaussian-shaped posterior.
 # Add LCDM and wCDM-plane points to protect nested model comparisons.
 while len(points)<a.n:
  x=CENTER+SCALE*rng.normal(size=8)*1.25
  if len(points)%6==0:x[6:]=[-1,0]
  elif len(points)%6==1:x[7]=0
  if logprior(x)==0 and x[6]+.75*x[7]<-.15:points.append(x)
 todo=[(i,x) for i,x in enumerate(points) if not (directory/f'point_{i:04d}.npz').exists()]
 start=time.time();done=0
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker) as pool:
  futs={pool.submit(evaluate,t):t[0] for t in todo}
  for f in as_completed(futs):
   i,ok=f.result();done+=1
   if done%16==0 or not ok:
    print('training',done,'/',len(todo),'last',i,'ok',ok,'elapsed',round(time.time()-start),flush=True)
 fit()
