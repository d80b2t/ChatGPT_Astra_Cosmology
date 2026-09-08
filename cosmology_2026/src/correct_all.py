"""Memory-efficient shared worker pool for multiple model correction runs."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time
import numpy as np
import correct as C
from model import ROOT,Likelihood

def main(models,n,workers):
 if n%4:raise ValueError('n must be divisible by four')
 tasks=[];nper=n//4
 for model in models:
  name=f'{model}_dovekie_ap';chain=np.load(ROOT/f'chains/{name}.npz')
  samples=chain['samples'];logp=chain['logp'];draws=samples.shape[1];rng=np.random.default_rng(7212026)
  bins=np.linspace(0,draws,nper+1).astype(int)
  for c in range(4):
   for j in range(nper):
    i=int(rng.integers(bins[j],bins[j+1]));tasks.append((model,name,c*nper+j,samples[c,i],logp[c,i],c,i))
 tasks.sort(key=lambda t:(t[2]%nper,t[2]//nper,t[0]))
 # Initialize CAMB separately in each worker rather than inheriting Fortran caches.
 C.L=None;start=time.time();done={m:0 for m in models}
 with ProcessPoolExecutor(max_workers=workers,initializer=C.init_worker) as pool:
  future_model={pool.submit(C.evaluate,t):t[0] for t in tasks}
  for f in as_completed(future_model):
   m=future_model[f];i,ok=f.result();done[m]+=1
   if done[m]%24==0 or not ok:print('exact',m,done[m],'/',n,'ok',ok,'elapsed',round(time.time()-start),flush=True)
   if done[m]==n:C.summarize(f'{m}_dovekie_ap')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('models',nargs='+',choices=['lcdm','wcdm','cpl']);p.add_argument('--n',type=int,default=800);p.add_argument('--workers',type=int,default=6);a=p.parse_args();main(a.models,a.n,a.workers)
