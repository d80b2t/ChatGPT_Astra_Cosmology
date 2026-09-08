"""Direct-CAMB refinement of surrogate modes; report best evaluated deviances.

One local gradient correction uses the surrogate Hessian as a preconditioner.
This is an approximate local profile, not a proof of a global optimum.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
import argparse,json
import numpy as np
from scipy.optimize import minimize
from model import ROOT,Likelihood,logprior
from emulator import Emulator
from sample import unpack,MODELS

L=None;E=None
def phi_emu(v,m):
 x,A,P=unpack(v,m)
 if not np.isfinite(logprior(x,A,P)):return 1e15
 return -L.loglike(E.one(x),A,P)

def calc(task):
 m,label,v=task;path=ROOT/f'results/bestfit/{m}_{label}.npz'
 if path.exists():
  z=np.load(path)
  if np.array_equal(z['v'],v):return dict(z)
 x,A,P=unpack(v,m);y,sn=L.theory(x,exact_sn=True)
 c=L.components(y,A,P,exact_sn=sn);phi=.5*sum(c.values())
 z=dict(v=v,x=x,A=A,P=P,y=y,phi=phi,delta=phi-phi_emu(v,m),sn=np.array([sn['dovekie'],sn['pantheon']]),components=np.array(list(c.values())),component_names=np.array(list(c)))
 np.savez_compressed(path,**z);return z

def main(workers):
 global L,E
 L=Likelihood();E=Emulator();out={}
 (ROOT/'results/bestfit').mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for m,n in MODELS.items():
   prop=np.load(ROOT/f'results/{m}_dovekie_ap_proposal.npz');v=prop['mean'];cov=prop['cov'];d=len(v)
   tasks=[(m,'start',v)]
   h=.2*np.sqrt(np.diag(cov))
   for i in range(n):
    step=np.eye(d)[i]*h[i]
    tasks.extend([(m,f'p{i}',v+step),(m,f'm{i}',v-step)])
   z=list(pool.map(calc,tasks));gd=np.zeros(d)
   for i in range(n):gd[i]=(float(z[1+2*i]['delta'])-float(z[2+2*i]['delta']))/(2*h[i])
   # Calibration changes do not require another Boltzmann calculation.
   y=z[0]['y'];sn=dict(zip(['dovekie','pantheon'],z[0]['sn']))
   for i in range(n,d):
    step=np.eye(d)[i]*h[i];f=[]
    for u in [v+step,v-step]:
     x,A,P=unpack(u,m);f.append(-L.loglike(y,A,P,exact_sn=sn)-phi_emu(u,m))
    gd[i]=(f[0]-f[1])/(2*h[i])
   obj=lambda u:phi_emu(u,m)+gd@(u-v)
   # The native SimAll table is discrete; derivative-free refinement is appropriate.
   opt=minimize(obj,v,method='Powell',options={'xtol':1e-4,'ftol':1e-10,'maxiter':200})
   step=opt.x-v;norm=float(np.sqrt(step@np.linalg.solve(cov,step)))
   if norm>1:step/=norm
   refined=calc((m,'refined',v+step));best=min([z[0],refined],key=lambda a:float(a['phi']))
   np.savez_compressed(ROOT/f'results/{m}_bestfit.npz',**best)
   out[m]={'deviance':float(2*best['phi']),'start_deviance':float(2*z[0]['phi']),'refined_deviance':float(2*refined['phi']),'refinement_step_posterior_sigma':norm,'optimizer_success':bool(opt.success),'components':dict(zip(best['component_names'].tolist(),best['components'].tolist())),'method':'Best evaluated direct-CAMB deviance after one local surrogate-preconditioned gradient refinement; approximate local profile'}
   print(m,out[m],flush=True)
 baseline=out['lcdm']['deviance']
 for m in MODELS:
  out[m]['delta_deviance']=out[m]['deviance']-baseline
  out[m]['delta_penalized_deviance']=out[m]['delta_deviance']+2*(MODELS[m]-6)
 (ROOT/'results/model_comparison.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.workers)
