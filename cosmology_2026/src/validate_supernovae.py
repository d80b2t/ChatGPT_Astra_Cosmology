"""Independent background-only check of released SN-only flat-LCDM constraints.

Gauss-Legendre quadrature is independent of CAMB's distance integrator. Radiation
is neglected for this low-redshift validation only, never for the main fits.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from model import Supernova,ROOT,ZGRID

nodes,weights=leggauss(96)
def mu(om,z,zhel):
 z=np.asarray(z);u=z[:,None]*(nodes+1)/2
 dc=(299792.458/70)*z/2*np.sum(weights/np.sqrt(om*(1+u)**3+1-om),axis=1)
 return 5*np.log10(dc*(1+zhel))+25

def main():
 out={};grid=np.linspace(.20,.46,1301)
 for name in ['dovekie','pantheon']:
  sn=Supernova(name)
  chi=np.array([sn.compressed_chi2(mu(o,ZGRID,ZGRID)) for o in grid])
  w=np.exp(-(chi-chi.min())/2);w/=w.sum()
  mean=float(grid@w);sd=float(np.sqrt((grid-mean)**2@w))
  best=grid[chi.argmin()];r=mu(best,sn.z,sn.zhel)-sn.mu
  r-=sn.p1@r/sn.c;exact=float(r@sn.P@r)
  out[name]={'Omega_m_mean':mean,'Omega_m_sd':sd,'Omega_m_mode':float(best),'chi2_mode':exact,'interpolation_chi2_error':float(chi.min()-exact),'n':sn.n,'method':'96-node Gauss-Legendre distances; flat LCDM without radiation; uncalibrated SN only'}
  np.savez_compressed(ROOT/f'results/{name}_sn_only_grid.npz',Omega_m=grid,chi2=chi,weights=w)
  print(name,out[name],flush=True)
 (ROOT/'results/supernova_validation.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
