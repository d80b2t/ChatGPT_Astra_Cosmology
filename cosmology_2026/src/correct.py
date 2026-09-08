"""Evaluate direct CAMB target likelihoods for stratified MCMC draws.

Final posterior summaries use self-normalized importance sampling, with the
original MCMC density retained explicitly. Surrogate chains are never relabeled
as direct Boltzmann-solver chains.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
import numpy as np
import arviz as az
from model import ROOT,Likelihood,NAMES
from sample import unpack,MODELS

L=None
def init_worker():
 global L
 if L is None:L=Likelihood()

def evaluate(task):
 model,name,i,v,lp,ci,di=task
 path=ROOT/f'results/exact/{name}_{i:05d}.npz'
 if path.exists():
  with np.load(path) as cached:
   if np.array_equal(cached['v'],v) and float(cached['logproposal'])==float(lp):return i,True
  raise RuntimeError(f'Exact cache selection changed: {path}; use a new output prefix')
 x,A,P=unpack(v,model)
 try:
  y,sn=L.theory(x,exact_sn=True)
  c=L.components(y,A,P,exact_sn=sn)
  cb=L.components(y,A,P,ap=False,exact_sn=sn)
  cp=L.components(y,A,P,sn='pantheon',exact_sn=sn)
  target=-.5*sum(c.values())
  np.savez_compressed(path,x=x,A=A,P=P,y=y,v=v,logproposal=lp,logtarget=target,
    logtarget_bao=-.5*sum(cb.values()),logtarget_pantheon=-.5*sum(cp.values()),
    logtarget_nosn=target+.5*sn['dovekie'],components=np.array(list(c.values())),
    component_names=np.array(list(c)),sn_exact=np.array([sn['dovekie'],sn['pantheon']]),chain=ci,draw=di)
  return i,True
 except Exception as e:
  path.with_suffix('.error').write_text(repr(e));return i,False

def wquant(x,w,ps=[.16,.5,.84]):
 idx=np.argsort(x);cdf=np.cumsum(w[idx])-.5*w[idx];cdf/=w.sum()
 return np.interp(ps,cdf,x[idx])

def summarize(name):
 files=sorted((ROOT/'results/exact').glob(name+'_*.npz'))
 f=[np.load(p) for p in files]
 x=np.array([z['x'] for z in f]);derived=np.array([z['y'][-5:] for z in f]);lp=np.array([float(z['logproposal']) for z in f])
 vals=np.column_stack([x, np.array([float(z['A']) for z in f]),np.array([float(z['P']) for z in f]),derived,x[:,6]-x[:,7]/4,derived[:,4]*np.sqrt(derived[:,1]/.3)])
 names=NAMES+['A_cmb','P_act','H0','Omega_m','r_drag','age_Gyr','sigma8','w0','S8']
 allinfo={}
 for variant,key in [('baseline','logtarget'),('bao_only','logtarget_bao'),('pantheon','logtarget_pantheon'),('no_sn','logtarget_nosn')]:
  target=np.array([float(z[key]) for z in f]);lw=target-lp;w=np.exp(lw-np.max(lw));w/=w.sum()
  _,k=az.psislw(lw)
  neff=1/np.sum(w*w)
  stats={}
  for i,n in enumerate(names):
   lo,med,hi=wquant(vals[:,i],w);mean=float(w@vals[:,i]);std=float(np.sqrt(w@((vals[:,i]-mean)**2)))
   mcse=float(np.sqrt(np.sum(w*w*(vals[:,i]-mean)**2)))
   stats[n]={'mean':mean,'sd':std,'q16':float(lo),'median':float(med),'q84':float(hi),'mcse_mean_iid':mcse}
  adequate=bool(neff>=400 and np.isfinite(k) and k<.7)
  info={'n_exact':len(f),'weight_ess':float(neff),'pareto_k':float(k),'max_weight':float(w.max()),'logweight_sd':float(lw.std()),'adequate_for_report':adequate,'quality_rule':'At least 400 weight ESS and Pareto k < 0.7; also inspect chain convergence and MCSE. This is a numerical diagnostic, not a proof of full posterior support.','parameters':stats}
  allinfo[variant]=info
  np.savez_compressed(ROOT/f'chains/{name}_{variant}_weighted.npz',parameters=vals,names=names,weights=w,logtarget=target,logproposal=lp,chain=[int(z['chain']) for z in f],draw=[int(z['draw']) for z in f],adequate_for_report=adequate,posterior_type='direct-CAMB self-normalized importance weights; check diagnostics before use')
  print(name,variant,'ESS',round(neff,1),'k',round(float(k),3),'H0',stats['H0']['mean'],stats['H0']['sd'],'w0,wa',stats['w0']['mean'],stats['wa']['mean'],flush=True)
 (ROOT/f'results/{name}_corrected.json').write_text(json.dumps(allinfo,indent=2))
 return allinfo

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);p.add_argument('--n',type=int,default=800);p.add_argument('--workers',type=int,default=8);p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
 if a.n%4:raise ValueError('n must be divisible by four')
 name=f'{a.model}_dovekie_ap'
 if a.summarize_only:summarize(name);raise SystemExit
 (ROOT/'results/exact').mkdir(exist_ok=True)
 chain=np.load(ROOT/f'chains/{name}.npz');samples=chain['samples'];lps=chain['logp'];nper=a.n//4;draws=samples.shape[1];rng=np.random.default_rng(7212026)
 tasks=[]
 for c in range(4):
  bins=np.linspace(0,draws,nper+1).astype(int)
  for j in range(nper):
   i=int(rng.integers(bins[j],bins[j+1]));tasks.append((a.model,name,len(tasks),samples[c,i],lps[c,i],c,i))
 # Interleave independent chains in the work queue; cached IDs stay unchanged.
 tasks.sort(key=lambda task:(task[2]%nper,task[2]//nper))
 start=time.time();done=0
 # Each worker initializes CAMB independently to avoid inherited Fortran caches.
 L=None
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker) as pool:
  futs=[pool.submit(evaluate,t) for t in tasks]
  for future in as_completed(futs):
   i,ok=future.result();done+=1
   if done%24==0 or not ok:print('exact',a.model,done,'/',len(tasks),'ok',ok,'elapsed',round(time.time()-start),flush=True)
 summarize(name)
