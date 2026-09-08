"""Numerical-accuracy checks at all fitted modes, with direct likelihood values."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np
from model import ROOT,Likelihood

def main():
 L=Likelihood();out={}
 for m in ['lcdm','wcdm','cpl']:
  f=np.load(ROOT/f'results/{m}_bestfit.npz');path=ROOT/f'results/{m}_high_accuracy.npz'
  if path.exists():
   h=np.load(path)
   if not np.array_equal(h['x'],f['x']):raise RuntimeError('Accuracy cache belongs to another fitted point')
   y=h['y'];sn=dict(zip(['dovekie','pantheon'],h['sn']))
  else:
   y,sn=L.theory(f['x'],accuracy='high',exact_sn=True)
   np.savez_compressed(path,x=f['x'],y=y,sn=np.array([sn['dovekie'],sn['pantheon']]))
  A=float(f['A']);P=float(f['P']);dy=y-f['y'];dp=dy[L.slices['planck']]/A**2
  da=dy[L.slices['act']]/A**2*np.repeat([1,1/P,1/P**2],L.act_counts)
  db=dy[L.slices['bao']]
  metric=float(dp@L.pl_inv@dp+da@L.act_inv@da+db@L.apinv@db)
  ch=L.components(y,A,P,exact_sn=sn);base=dict(zip(f['component_names'],f['components']))
  delta={k:float(ch[k]-base[k]) for k in ch}
  out[m]={'data_space_squared_difference':metric,'data_space_norm':float(np.sqrt(metric)),'delta_deviance':float(sum(delta.values())),'delta_components':delta,'derived_change':(y[-5:]-f['y'][-5:]).tolist()}
  print(m,out[m],flush=True)
 baseline=out['lcdm']['delta_deviance']
 for m in out:out[m]['change_in_model_deviance_difference']=out[m]['delta_deviance']-baseline
 (ROOT/'results/numerical_accuracy.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
