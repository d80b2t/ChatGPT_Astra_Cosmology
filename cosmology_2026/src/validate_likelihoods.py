"""Cross-check manual CMB assembly against pinned upstream likelihood classes."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
import numpy as np
import camb
from model import ROOT,CMB,Likelihood,FID,parameters
sys.path.insert(0,str(ROOT/'vendor'))
from act_dr6_cmbonly import ACTDR6CMBonly
from PlanckActCut import PlanckActCut

def main():
 L=Likelihood();directory=ROOT/'packages/data/ACTDR6CMBonly'
 if not directory.exists():directory.symlink_to(CMB/'ACTDR6CMBonly',target_is_directory=True)
 act=ACTDR6CMBonly({},packages_path=str(ROOT/'packages'))
 pl=PlanckActCut({'path':str(CMB),'dataset_file':'plik_lite_v22.dataset','dataset_params':{'use_cl':'tt te ee','lmin_cuts':'0 0 0','lmax_cuts':'1000 600 600'}},packages_path=str(ROOT/'packages'))
 r=camb.get_results(parameters(FID));cls=r.get_total_cls(lmax=8501,CMB_unit='muK');cl={'tt':cls[:,0],'te':cls[:,3],'ee':cls[:,1]}
 y=np.load(ROOT/'results/fiducial.npz')['y'];out=[]
 for A,P in [(1.,1.),(1.002,.995),(.997,1.007)]:
  manual=L.components(y,A,P);ca=act.chi_square(cl,A,P);cp=pl.get_chi_squared(0,cl['tt'],cl['te'],cl['ee'],A)
  row={'A':A,'P':P,'manual_act':manual['act'],'upstream_act':float(ca),'act_difference':float(ca-manual['act']),'manual_planck':manual['planck_high'],'upstream_planck':float(cp),'planck_difference':float(cp-manual['planck_high'])}
  print(row,flush=True);out.append(row)
  assert abs(row['act_difference'])<1e-6
  # Upstream Planck code gives excluded bins variance 1e10, rather than removing them.
  assert abs(row['planck_difference'])<1e-4
 (ROOT/'results/upstream_likelihood_validation.json').write_text(json.dumps({'checks':out,'interpretation':'ACT matches directly. Any tiny positive Planck difference comes from the upstream variance-1e10 treatment of excluded bins, which our likelihood removes.'},indent=2))
if __name__=='__main__':main()
