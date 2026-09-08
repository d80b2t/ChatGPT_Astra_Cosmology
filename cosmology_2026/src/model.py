"""Real-data likelihoods and direct CAMB predictions. No mock observations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
import camb, sacc
from cobaya.likelihoods.planck_2018_highl_plik.TTTEEE_lite_native import TTTEEE_lite_native
from cobaya.likelihoods.planck_2018_lowl.TT import TT
from cobaya.likelihoods.planck_2018_lowl.EE import EE

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; CMB=DATA/'cmb'
# Spectrum coordinates: physical densities, 100 theta_star, primordial spectrum,
# optical depth, and w(z=1/3), wa. w0 = wp - wa/4.
NAMES=['ombh2','omch2','theta100','logA','ns','tau','wp','wa']
CENTER=np.array([.0225,.119,1.0411,3.045,.973,.055,-.975,-.5])
SCALE=np.array([.00024,.0022,.00050,.034,.007,.013,.060,.65])
FID=CENTER.copy(); FID[6:]=[-1,0]
ZGRID=np.geomspace(.005,2.5,100)

class Supernova:
 def __init__(self,name):
  self.name=name
  if name=='dovekie':
   # Despite its .csv extension, the pinned release is SNANA whitespace format.
   d=pd.read_csv(DATA/'dovekie.csv',sep=r'\s+',comment='#')
   f=np.load(DATA/'dovekie_precision.npz'); n=int(f[f.files[0]][0])
   P=np.zeros((n,n)); P[np.triu_indices(n)]=f[f.files[1]]
   P=P+np.triu(P,1).T
   assert len(d)==n and np.all(d.zHD>0)
   mu=d.MU.to_numpy();self.z=d.zHD.to_numpy();self.zhel=d.zHEL.to_numpy()
  elif name=='pantheon':
   d=pd.read_csv(DATA/'pantheon_plus.dat',sep=r'\s+')
   c=np.loadtxt(DATA/'pantheon_plus.cov');n=int(c[0]);c=c[1:].reshape(n,n)
   mask=d.zHD.to_numpy()>.01
   c=c[np.ix_(mask,mask)];P=cho_solve(cho_factor(c,lower=True),np.eye(mask.sum()))
   d=d[mask];mu=d.m_b_corr.to_numpy();self.z=d.zHD.to_numpy();self.zhel=d.zHEL.to_numpy()
  else: raise ValueError(name)
  self.n=len(mu);self.mu=mu;self.P=P
  self.p1=P.sum(axis=1);self.c=float(self.p1.sum())
  self.Pm=P-np.outer(self.p1,self.p1)/self.c
  self.B=CubicSpline(np.log(ZGRID),np.eye(len(ZGRID)),axis=0)(np.log(self.z))
  self.K=self.B.T@self.Pm@self.B
  # Reference cosmology only improves numerical conditioning, no prior is added.
  p=parameters(FID,accuracy='background');bg=camb.get_background(p)
  self.fidmu=5*np.log10(bg.angular_diameter_distance(self.z)*(1+self.z)*(1+self.zhel))+25
  residual=self.fidmu-mu
  self.b=self.B.T@self.Pm@residual
  self.c0=float(residual@self.Pm@residual)
  self.fidgrid=5*np.log10(bg.angular_diameter_distance(ZGRID)*(1+ZGRID)**2)+25
 def compressed_chi2(self,mugrid):
  d=mugrid-self.fidgrid
  return float(self.c0+2*self.b@d+d@self.K@d)
 def exact_chi2(self,bg):
  m=5*np.log10(bg.angular_diameter_distance(self.z)*(1+self.z)*(1+self.zhel))+25
  d=m-self.mu
  # Offset subtraction first avoids catastrophic cancellation for m_B data.
  d=d-(self.p1@d)/self.c
  return float(d@self.P@d)

def parameters(x,accuracy='standard'):
 wb,wc,theta,logA,ns,tau,wp,wa=np.asarray(x)
 w0=wp-wa/4
 p=camb.CAMBparams()
 p.set_dark_energy(w=w0,wa=wa,dark_energy_model='ppf')
 p.set_cosmology(thetastar=theta/100,ombh2=wb,omch2=wc,mnu=.06,
                 nnu=3.046,num_massive_neutrinos=1,tau=tau,theta_H0_range=(35,100))
 p.InitPower.set_params(As=np.exp(logA)*1e-10,ns=ns)
 if accuracy!='background':
  p.set_for_lmax(8501,lens_potential_accuracy=1,lens_output_margin=700)
  p.set_accuracy(AccuracyBoost=1.2,lAccuracyBoost=1.2,lSampleBoost=1.2,min_l_logl_sampling=6000)
  p.set_matter_power(redshifts=[0],kmax=5.0,silent=True)
  p.NonLinear=camb.model.NonLinear_both
  if accuracy=='high':
   p.set_for_lmax(8501,lens_potential_accuracy=2,lens_output_margin=1200)
   p.set_accuracy(AccuracyBoost=2,lAccuracyBoost=2,lSampleBoost=2)
 return p

class Likelihood:
 def __init__(self,with_pantheon=True):
  logging.getLogger('cobaya').setLevel(logging.ERROR)
  # The public release archives are flat; provide Cobaya's expected layout.
  for cls in [TTTEEE_lite_native,TT,EE]:
   directory=Path(cls.get_path(str(ROOT/'packages')))
   directory.parent.mkdir(parents=True,exist_ok=True)
   if not directory.exists():
    try:directory.symlink_to(CMB,target_is_directory=True)
    except FileExistsError:pass
  # Atomic installation marker: concurrent workers must never see a truncated file.
  if not (CMB/'version.dat').exists():
   marker=CMB/f'version.{os.getpid()}.tmp';marker.write_text('1')
   marker.replace(CMB/'version.dat')
  pl=TTTEEE_lite_native({'path':str(CMB)},packages_path=str(ROOT/'packages'))
  offsets=np.cumsum([0]+[len(b) for b in pl.used_bins])
  inds=[]; windows=[];self.pl_ell=[];self.pl_counts=[]
  for j,cut in enumerate([1000,600,600]):
   ids=pl.used_bins[j]
   keep=ids[pl.blmax[ids]<=cut]
   self.pl_ell.append(pl.lav[keep]);self.pl_counts.append(len(keep))
   inds.extend(keep+offsets[j]);w=np.zeros((len(keep),8502))
   for k,i in enumerate(keep):w[k,pl.blmin[i]:pl.blmax[i]+1]=pl.weights[pl.blmin[i]:pl.blmax[i]+1]
   windows.append(csr_matrix(w))
  inds=np.asarray(inds,dtype=int)
  self.pl_windows=windows
  self.pl_data=pl.X_data[inds]
  self.pl_cov=pl.cov[np.ix_(inds,inds)]
  self.pl_inv=np.linalg.inv(self.pl_cov)
  self.npl=len(inds)
  s=sacc.Sacc.load_fits(str(CMB/'ACTDR6CMBonly/v1.0/dr6_data_cmbonly.fits'))
  self.act_windows=[];self.act_counts=[];ainds=[];self.act_ell=[]
  for dt in ['cl_00','cl_0e','cl_ee']:
   [(t1,t2)]=s.get_tracer_combinations(dt)
   ell,d,ind=s.get_ell_cl(dt,t1,t2,return_ind=True)
   keep=(ell>=600)&(ell<=6500);ind=ind[keep]
   w=s.get_bandpower_windows(ind);matrix=np.zeros((len(ind),8502))
   matrix[:,w.values.astype(int)]=w.weight.T
   self.act_windows.append(csr_matrix(matrix));self.act_counts.append(len(ind));ainds.extend(ind);self.act_ell.append(ell[keep])
  self.act_data=s.mean[ainds]
  self.act_cov=s.covariance.covmat[np.ix_(ainds,ainds)]
  self.act_inv=np.linalg.inv(self.act_cov);self.nact=len(ainds)
  self.lowT=TT({},packages_path=str(ROOT/'packages'))
  self.lowE=EE({},packages_path=str(ROOT/'packages'))
  self.sn={name:Supernova(name) for name in (['dovekie','pantheon'] if with_pantheon else ['dovekie'])}
  d=np.loadtxt(DATA/'desi_dr2_mean.txt',dtype=str,skiprows=1)
  self.baoz=d[:,0].astype(float);self.baoy=d[:,1].astype(float);self.baotype=d[:,2]
  self.baocov=np.loadtxt(DATA/'desi_dr2_cov.txt');self.baoinv=np.linalg.inv(self.baocov)
  # Replace (do not append) the two Ly-alpha BAO-only entries by Eq. 26 of
  # DESI DR2 IV, arXiv:2607.27410v3: ordering in original vector is DH, DM.
  self.apy=self.baoy.copy();self.apcov=self.baocov.copy()
  idx=np.where(self.baoz==2.33)[0];assert list(self.baotype[idx])==['DH_over_rs','DM_over_rs']
  other=np.setdiff1d(np.arange(len(self.baoy)),idx)
  assert np.allclose(self.apcov[np.ix_(idx,other)],0)
  self.apy[idx]=[8.600,39.32]
  self.apcov[np.ix_(idx,idx)]=[[.066**2,.225*.066*.33],[.225*.066*.33,.33**2]]
  self.apinv=np.linalg.inv(self.apcov)
  k=0;self.slices={}
  for name,n in [('planck',self.npl),('act',self.nact),('lowT',28),('lowE',28),('bao',13),('sn',len(ZGRID)),('derived',5)]:
   self.slices[name]=slice(k,k+n);k+=n
  self.nout=k

 def theory(self,x,accuracy='standard',exact_sn=False):
  p=parameters(x,accuracy=accuracy);r=camb.get_results(p)
  cls=r.get_total_cls(lmax=8501,CMB_unit='muK')
  spec=[cls[:,0],cls[:,3],cls[:,1]]
  planck=np.concatenate([w@a for w,a in zip(self.pl_windows,spec)])
  act=np.concatenate([w@a for w,a in zip(self.act_windows,spec)])
  d=r.get_derived_params();rd=d['rdrag']
  DM=r.angular_diameter_distance(self.baoz)*(1+self.baoz);DH=299792.458/r.hubble_parameter(self.baoz)
  bao=np.array([{'DM_over_rs':m/rd,'DH_over_rs':h/rd,'DV_over_rs':(z*m*m*h)**(1/3)/rd}[t] for m,h,z,t in zip(DM,DH,self.baoz,self.baotype)])
  sn=5*np.log10(r.angular_diameter_distance(ZGRID)*(1+ZGRID)**2)+25
  om=(p.ombh2+p.omch2+p.omnuh2)/(p.H0/100)**2
  derived=[p.H0,om,rd,d['age'],float(r.get_sigma8()[0])]
  y=np.concatenate([planck,act,cls[2:30,0],cls[2:30,1],bao,sn,derived])
  if exact_sn:
   return y,{k:v.exact_chi2(r) for k,v in self.sn.items()}
  return y

 def components(self,y,A=1.0,P=1.0,sn='dovekie',ap=True,exact_sn=None,planck_only=False):
  sl=self.slices
  a=y[sl['planck']]/A**2-self.pl_data
  out={'planck_high':float(a@self.pl_inv@a)}
  if not planck_only:
   cal=np.repeat([1,1/P,1/P**2],self.act_counts)
   a=y[sl['act']]*cal/A**2-self.act_data
   out['act']=float(a@self.act_inv@a)
  low=np.r_[0,0,y[sl['lowT']]];out['planck_lowT']=-2*self.lowT.log_likelihood(low,A)
  low=np.r_[0,0,y[sl['lowE']]];out['planck_lowE']=-2*self.lowE.log_likelihood(low,A)
  a=y[sl['bao']]-(self.apy if ap else self.baoy)
  out['bao']=float(a@(self.apinv if ap else self.baoinv)@a)
  if sn:
   out['sn']=self.sn[sn].compressed_chi2(y[sl['sn']]) if exact_sn is None else exact_sn[sn]
  out['calibration_prior']=((A-1)/.0025)**2
  return out

 def loglike(self,y,A=1,P=1,**kwargs):
  return -.5*sum(self.components(y,A,P,**kwargs).values())

def logprior(x,A=1,P=1):
 wb,wc,th,la,ns,tau,wp,wa=x;w0=wp-wa/4
 vals=[(.0205,wb,.0245),(.105,wc,.135),(1.038,th,1.044),(2.8,la,3.3),(.92,ns,1.03),(.01,tau,.12),(-2,w0,0),(-3,wa,1),(.98,A,1.02),(.9,P,1.1)]
 if not all(a<b<c for a,b,c in vals) or w0+wa>=0: return -np.inf
 return 0.0

if __name__=='__main__':
 import time,json
 t=time.time();L=Likelihood();print('init seconds',time.time()-t,'dimensions',L.npl,L.nact,{k:v.n for k,v in L.sn.items()},flush=True)
 t=time.time();y,sn=L.theory(FID,exact_sn=True)
 print('theory seconds',time.time()-t,'derived',y[L.slices['derived']],flush=True)
 print('chi2',L.components(y,exact_sn=sn),flush=True)
 print('SN interpolation chi2 error',{k:L.sn[k].compressed_chi2(y[L.slices['sn']])-v for k,v in sn.items()},flush=True)
 np.savez(ROOT/'results/fiducial.npz',x=FID,y=y)
