"""Four independent adaptive-warmup Metropolis chains; frozen production kernel."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
import argparse,json,time
import numpy as np
from scipy.optimize import minimize
import arviz as az
from model import Likelihood,ROOT,CENTER,SCALE,FID,NAMES,logprior
from emulator import Emulator

MODELS={'lcdm':6,'wcdm':7,'cpl':8}
def unpack(v,model):
 n=MODELS[model];x=FID.copy();x[:n]=CENTER[:n]+SCALE[:n]*np.asarray(v[:n])
 return x,1+.0025*v[n],1+.02*v[n+1]
def pack(x,A,P,model):
 n=MODELS[model];return np.r_[(x[:n]-CENTER[:n])/SCALE[:n],(A-1)/.0025,(P-1)/.02]

class Posterior:
 def __init__(self,model,sn='dovekie',ap=True):
  self.model=model;self.sn=sn;self.ap=ap;self.L=Likelihood();self.E=Emulator()
 def __call__(self,v):
  x,A,P=unpack(v,self.model)
  if not np.isfinite(logprior(x,A,P)):return -np.inf
  y=self.E.one(x)
  if not (35<y[-5]<100 and .05<y[-4]<.8) or np.any(y[self.L.slices['lowE']]<0):return -np.inf
  lp=self.L.loglike(y,A,P,sn=self.sn,ap=self.ap)
  return lp if np.isfinite(lp) else -np.inf

def proposal_fit(post,model):
 x=FID.copy()
 if model=='cpl':x[6:]=[-.99,-.7]
 v=pack(x,1,1,model)
 def obj(v):
  lp=post(v);return -lp if np.isfinite(lp) else 1e12
 opt=minimize(obj,v,method='Powell',options={'maxiter':100,'xtol':.0001,'ftol':1e-8})
 v=opt.x;d=len(v);h=.12;H=np.zeros((d,d));f=obj(v)
 for i in range(d):
  a=np.zeros(d);a[i]=h
  H[i,i]=(obj(v+a)+obj(v-a)-2*f)/h**2
  for j in range(i):
   b=np.zeros(d);b[j]=h
   H[i,j]=H[j,i]=(obj(v+a+b)-obj(v+a-b)-obj(v-a+b)+obj(v-a-b))/(4*h*h)
 ev,U=np.linalg.eigh(H);cov=(U/np.maximum(ev,.05))@U.T
 return v,cov,{'success':bool(opt.success),'message':str(opt.message),'chi2':2*f,'hessian_eigenvalues':ev.tolist()}

def run_chain(task):
 model,sn,ap,chain_id,mean,cov,warm,draws=task
 post=Posterior(model,sn,ap);rng=np.random.default_rng(20260904+chain_id*1000+MODELS[model]*100)
 d=len(mean);C=np.linalg.cholesky(cov);scale=2.38/np.sqrt(d)
 while True:
  v=mean+C@rng.normal(size=d)*1.8;lp=post(v)
  if np.isfinite(lp):break
 hist=[];accept=0;blockacc=0
 for i in range(warm):
  prop=v+scale*C@rng.normal(size=d);lnew=post(prop)
  if np.log(rng.random())<lnew-lp:v=prop;lp=lnew;accept+=1;blockacc+=1
  hist.append(v.copy())
  if (i+1)%500==0:
   rate=blockacc/500;blockacc=0
   scale*=np.exp(np.clip(rate-.25,-.2,.2))
   if i>=999:
    emp=np.cov(np.asarray(hist)[len(hist)//3:].T)
    C=np.linalg.cholesky(.9*emp+.1*cov+1e-6*np.eye(d))
 out=np.zeros((draws,d));lps=np.zeros(draws);accepted=0
 # No proposal adaptation after warmup.
 for i in range(draws):
  prop=v+scale*C@rng.normal(size=d);lnew=post(prop)
  if np.log(rng.random())<lnew-lp:v=prop;lp=lnew;accepted+=1
  out[i]=v;lps[i]=lp
 name=f'{model}_{sn or "nosn"}_{"ap" if ap else "bao"}'
 np.savez_compressed(ROOT/f'chains/{name}_{chain_id}.npz',samples=out,logp=lps,seed=20260904+chain_id*1000+MODELS[model]*100,acceptance=accepted/draws,warmup=warm,proposal_chol=C,proposal_scale=scale)
 return chain_id,out,lps,accepted/draws

def main(a):
 name=f'{a.model}_{a.sn or "nosn"}_{"bao" if a.bao_only else "ap"}'
 post=Posterior(a.model,a.sn,not a.bao_only)
 t=time.time();mean,cov,info=proposal_fit(post,a.model)
 print('proposal',name,info,'seconds',round(time.time()-t),flush=True)
 np.savez(ROOT/f'results/{name}_proposal.npz',mean=mean,cov=cov)
 tasks=[(a.model,a.sn,not a.bao_only,i,mean,cov,a.warmup,a.draws) for i in range(4)]
 with ProcessPoolExecutor(max_workers=4) as pool:res=list(pool.map(run_chain,tasks))
 res.sort();samples=np.stack([r[1] for r in res]);logp=np.stack([r[2] for r in res])
 n=MODELS[a.model];names=NAMES[:n]+['A_cmb','P_act']
 idata=az.from_dict(posterior={k:samples[:,:,i] for i,k in enumerate(names)})
 diag=az.summary(idata,kind='diagnostics',round_to=6)
 diag.to_csv(ROOT/f'results/{name}_diagnostics.csv')
 info.update({'acceptance':[r[3] for r in res],'seconds':time.time()-t,'draws_per_chain':a.draws,'warmup_per_chain':a.warmup,'max_rhat':float(diag.r_hat.max()),'min_ess_bulk':float(diag.ess_bulk.min()),'min_ess_tail':float(diag.ess_tail.min())})
 np.savez_compressed(ROOT/f'chains/{name}.npz',samples=samples,logp=logp,names=names)
 (ROOT/f'results/{name}_sampling.json').write_text(json.dumps(info,indent=2))
 print(name,info,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);p.add_argument('--sn',default='dovekie');p.add_argument('--bao-only',action='store_true');p.add_argument('--warmup',type=int,default=5000);p.add_argument('--draws',type=int,default=30000);a=p.parse_args()
 if a.sn=='none':a.sn=None
 main(a)
