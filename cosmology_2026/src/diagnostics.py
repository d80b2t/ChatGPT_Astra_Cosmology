"""Physical-coordinate MCMC diagnostics and correlation-aware correction MCSE."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
import numpy as np
import arviz as az
from model import ROOT,CENTER,SCALE,FID,NAMES
from sample import MODELS
from emulator import Emulator

def raw_diagnostics():
 E=Emulator()
 for m,n in MODELS.items():
  name=f'{m}_dovekie_ap';p=ROOT/f'chains/{name}.npz';z=np.load(p)
  v=z['samples'];x=np.broadcast_to(FID,(*v.shape[:2],8)).copy()
  x[:,:,:n]=CENTER[:n]+SCALE[:n]*v[:,:,:n]
  derived=[];flat=x.reshape(-1,8)
  for j in range(0,len(flat),5000):derived.append(E(flat[j:j+5000])[:,-5:])
  derived=np.concatenate(derived).reshape(*v.shape[:2],5)
  post={k:x[:,:,i] for i,k in enumerate(NAMES[:n])}
  post['A_cmb']=1+.0025*v[:,:,n];post['P_act']=1+.02*v[:,:,n+1]
  for i,k in enumerate(['H0','Omega_m','r_drag','age_Gyr','sigma8']):post[k]=derived[:,:,i]
  post['S8']=derived[:,:,4]*np.sqrt(derived[:,:,1]/.3)
  if m=='cpl':post['w0']=x[:,:,6]-x[:,:,7]/4
  diag=az.summary(az.from_dict(posterior=post),kind='diagnostics',round_to=6)
  diag.to_csv(ROOT/f'results/{name}_diagnostics.csv')
  infopath=ROOT/f'results/{name}_sampling.json';info=json.loads(infopath.read_text())
  info.update(max_rhat=float(diag.r_hat.max()),min_ess_bulk=float(diag.ess_bulk.min()),min_ess_tail=float(diag.ess_tail.min()),diagnostic_coordinates='physical; derived coordinates use the surrogate; all production draws')
  infopath.write_text(json.dumps(info,indent=2))
  np.savez_compressed(ROOT/f'results/{name}_trace.npz',**{k:a[:,::30] for k,a in post.items()},draw=np.arange(v.shape[1])[::30])
  # Save explicit metadata without changing any sampled value or log density.
  np.savez_compressed(p,samples=v,logp=z['logp'],names=z['names'],coordinate_system='standardized; use sample.unpack()',center=CENTER,scale=SCALE,model=m,posterior_type='surrogate',emulator_file='results/emulator.npz')
  print(name,'Rhat',info['max_rhat'],'ESS bulk',info['min_ess_bulk'],flush=True)

def weighted_diagnostics():
 allout={}
 for m in MODELS:
  name=f'{m}_dovekie_ap';variants={};base_influence=None;base_mean=None
  for variant in ['baseline','bao_only','pantheon','no_sn']:
   p=ROOT/f'chains/{name}_{variant}_weighted.npz'
   if not p.exists():continue
   f=np.load(p);v=f['parameters'];w=f['weights'];names=f['names'].tolist();N=len(v)
   mean=w@v;influence=(w*N)[:,None]*(v-mean)
   batches=[];cmeans=[]
   for c in range(4):
    ix=np.where(f['chain']==c)[0];ix=ix[np.argsort(f['draw'][ix])]
    cmeans.append((w[ix]@v[ix])/w[ix].sum())
    for group in np.array_split(ix,10):batches.append(influence[group].mean(axis=0))
   batches=np.array(batches);mcse=np.std(batches,axis=0,ddof=1)/np.sqrt(len(batches))
   sd=np.sqrt(w@((v-mean)**2));ratio=np.divide(mcse,sd,out=np.zeros_like(sd),where=sd>1e-12)
   variants[variant]={'n_batches':len(batches),'draws_per_batch_per_chain':N//40,'method':'40 contiguous batches of ordered stratified draws; SNIS influence function; uncertainty estimate retains short-range correlation',
     'mcse_mean':dict(zip(names,mcse.tolist())),'max_mcse_over_posterior_sd':float(ratio.max()),'chain_weighted_means':dict(zip(names,np.array(cmeans).T.tolist()))}
   if variant=='baseline':base_influence=influence;base_mean=mean
   elif base_influence is not None:
    diff=influence-base_influence;db=[]
    for c in range(4):
     ix=np.where(f['chain']==c)[0];ix=ix[np.argsort(f['draw'][ix])]
     for group in np.array_split(ix,10):db.append(diff[group].mean(axis=0))
    dm=np.std(db,axis=0,ddof=1)/np.sqrt(len(db))
    variants[variant]['mean_shift_from_baseline']=dict(zip(names,(mean-base_mean).tolist()))
    variants[variant]['mcse_of_paired_shift']=dict(zip(names,dm.tolist()))
   print(name,variant,'max MCSE / sd',round(float(ratio.max()),3),flush=True)
  allout[m]=variants
 (ROOT/'results/weighted_diagnostics.json').write_text(json.dumps(allout,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--weighted',action='store_true');a=p.parse_args()
 weighted_diagnostics() if a.weighted else raw_diagnostics()
