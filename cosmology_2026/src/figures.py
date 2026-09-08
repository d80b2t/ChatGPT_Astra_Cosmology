"""Publication figures from saved direct-CAMB results and actual observations."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import camb
from model import ROOT,Likelihood,parameters
from correct import wquant

COL={'lcdm':'#0072B2','wcdm':'#009E73','cpl':'#D55E00'}
LAB={'lcdm':r'$\Lambda$CDM','wcdm':r'$w$CDM','cpl':'CPL'}
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':10,
 'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,
 'lines.linewidth':1.2,'savefig.dpi':200,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
 fig.savefig(ROOT/f'paper/{name}.pdf',bbox_inches='tight',pad_inches=.035)
 fig.savefig(ROOT/f'paper/{name}.png',bbox_inches='tight',pad_inches=.035)
 plt.close(fig)
def weighted(model,variant='baseline'):
 f=np.load(ROOT/f'chains/{model}_dovekie_ap_{variant}_weighted.npz')
 return {k:f['parameters'][:,i] for i,k in enumerate(f['names'])},f['weights']
def contour(ax,x,y,w,color,bounds,alpha=.12,ls='-'):
 gx=np.linspace(*bounds[0],160);gy=np.linspace(*bounds[1],160);xx,yy=np.meshgrid(gx,gy)
 kde=gaussian_kde(np.vstack([x,y]),weights=w);kde.set_bandwidth(kde.factor*.85)
 z=kde(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
 zs=np.sort(z.ravel())[::-1];cs=np.cumsum(zs);cs/=cs[-1]
 lev=np.sort([zs[np.searchsorted(cs,p)] for p in [.68,.95]])
 ax.contourf(xx,yy,z,levels=[lev[0],z.max()*1.01],colors=[color],alpha=alpha)
 ax.contour(xx,yy,z,levels=lev,colors=color,linestyles=ls,linewidths=[.9,1.3])
 ax.set_xlim(*bounds[0]);ax.set_ylim(*bounds[1])

def posterior():
 data={m:weighted(m) for m in COL};fig,axs=plt.subplots(1,3,figsize=(7.1,2.5),layout='constrained')
 for ax,(xn,yn,xlab,ylab) in zip(axs,[('H0','Omega_m',r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]',r'$\Omega_m$'),('H0','S8',r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]',r'$S_8$'),('w0','wa',r'$w_0$',r'$w_a$')]):
  models=['cpl'] if xn=='w0' else list(COL)
  xx=np.concatenate([data[m][0][xn] for m in models]);yy=np.concatenate([data[m][0][yn] for m in models])
  bounds=[]
  for a in [xx,yy]:
   lo,hi=np.quantile(a,[.001,.999]);margin=(hi-lo)*.13;bounds.append([lo-margin,hi+margin])
  if xn=='w0':bounds[0][0]=min(bounds[0][0],-1.04);bounds[1][1]=max(bounds[1][1],.15)
  for m in models:
   d,w=data[m];contour(ax,d[xn],d[yn],w,COL[m],bounds)
  if xn=='w0':
   ax.plot(-1,0,'k+',ms=8,mew=1.3);ax.axvline(-1,color='.6',lw=.5,ls=':');ax.axhline(0,color='.6',lw=.5,ls=':')
   ax.text(.96,.96,r'$\Lambda$CDM reference',ha='right',va='top',transform=ax.transAxes,fontsize=7)
  ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.tick_params(direction='in',top=True,right=True)
 axs[0].legend([Line2D([0],[0],color=COL[m]) for m in COL],[LAB[m] for m in COL],loc='lower left',frameon=False)
 save(fig,'posterior')

def cmb(L):
 f=np.load(ROOT/'results/lcdm_bestfit.npz');y=f['y'];A=float(f['A']);P=float(f['P'])
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.25),layout='constrained')
 offsets_pl=np.cumsum([0]+L.pl_counts);offsets_act=np.cumsum([0]+L.act_counts)
 for j,(ax,title) in enumerate(zip(axs,['TT','TE','EE'])):
  ip=slice(offsets_pl[j],offsets_pl[j+1]);ia=slice(offsets_act[j],offsets_act[j+1])
  rp=(L.pl_data[ip]-y[L.slices['planck']][ip]/A**2)/np.sqrt(np.diag(L.pl_cov)[ip])
  ra=(L.act_data[ia]-y[L.slices['act']][ia]/A**2/P**j)/np.sqrt(np.diag(L.act_cov)[ia])
  ax.axhspan(-2,2,color='.95',zorder=0);ax.axhline(0,color='.35',lw=.6)
  ax.plot(L.pl_ell[j],rp,'o',ms=2.0,color=COL['lcdm'],label='Planck')
  ax.plot(L.act_ell[j],ra,'s',ms=2.8,color=COL['cpl'],label='ACT DR6')
  ax.set_title(title,fontsize=10);ax.set_xlabel(r'Multipole $\ell$');ax.set_xscale('log')
  ax.set_xticks([100,1000,6000],[r'$10^2$',r'$10^3$',r'$6000$']);ax.set_xlim(25,7000);ax.set_ylim(-4.3,4.3)
  ax.tick_params(direction='in',top=True,right=True)
 axs[0].set_ylabel(r'$(D_b^{\rm obs}-D_b^{\rm model})/\sigma_b$');axs[1].legend(frameon=False,loc='upper left')
 save(fig,'cmb_residuals')

def distances(L):
 best={m:np.load(ROOT/f'results/{m}_bestfit.npz') for m in COL}
 bg={m:camb.get_background(parameters(best[m]['x'],accuracy='background')) for m in COL}
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.45),layout='constrained')
 ax=axs[0];ref=best['lcdm']['y'][L.slices['bao']]
 for t,label,co,mark in [('DM_over_rs',r'$D_M/r_d$',COL['lcdm'],'o'),('DH_over_rs',r'$D_H/r_d$',COL['cpl'],'s'),('DV_over_rs',r'$D_V/r_d$',COL['wcdm'],'^')]:
  ix=L.baotype==t
  ax.errorbar(L.baoz[ix],100*(L.apy[ix]/ref[ix]-1),yerr=100*np.sqrt(np.diag(L.apcov)[ix])/ref[ix],fmt=mark,ms=3,color=co,lw=.7,capsize=1.5,label=label)
 ax.axhline(0,color='.3',lw=.7);ax.set_xlabel(r'Redshift $z$');ax.set_ylabel(r'BAO residual [per cent]');ax.legend(frameon=False,fontsize=7,loc='best')
 # Generalized least-squares bins, including the fitted common-offset projection.
 sn=L.sn['dovekie'];edges=np.unique(np.r_[sn.z.min()*.999,[.04,.08,.12,.2,.3,.4,.5,.6,.7,.8,.9,1.05],sn.z.max()*1.001])
 edges=edges[(edges>=sn.z.min()*.999)&(edges<=sn.z.max()*1.001)]
 ib=np.clip(np.searchsorted(edges,sn.z,side='right')-1,0,len(edges)-2)
 occupied=np.unique(ib);B=np.column_stack([ib==i for i in occupied]).astype(float)
 F=B.T@sn.P@B;Cb=np.linalg.inv(F);W=Cb@B.T@sn.P
 mu={m:5*np.log10(bg[m].angular_diameter_distance(sn.z)*(1+sn.z)*(1+sn.zhel))+25 for m in ['lcdm','cpl']}
 r=sn.mu-mu['lcdm'];r-=sn.p1@r/sn.c
 b=W@r;Cb-=np.ones_like(Cb)/sn.c;zb=np.array([np.exp(np.mean(np.log(sn.z[ib==i]))) for i in occupied])
 dm=mu['cpl']-mu['lcdm'];dm-=sn.p1@dm/sn.c
 ax=axs[1];ax.axhline(0,color='.3',lw=.7);ax.errorbar(zb,b,yerr=np.sqrt(np.maximum(np.diag(Cb),0)),fmt='o',ms=3,lw=.7,capsize=1.5,color=COL['lcdm'],label='Dovekie')
 ax.plot(zb,W@dm,color=COL['cpl'],label='CPL fit');ax.set_xscale('log');ax.set_xlabel(r'Redshift $z_{\rm HD}$');ax.set_ylabel(r'SN residual [mag]');ax.legend(frameon=False,fontsize=7)
 np.savez_compressed(ROOT/'results/sn_display_bins.npz',redshift=zb,residual=b,covariance=Cb,edges=edges,counts=B.sum(axis=0),cpl_prediction=W@dm)
 ax=axs[2];d,w=weighted('cpl');z=np.linspace(0,2.5,160);wz=d['w0'][:,None]+d['wa'][:,None]*z/(1+z)
 q=np.array([wquant(a,w,[.025,.16,.5,.84,.975]) for a in wz.T])
 ax.fill_between(z,q[:,0],q[:,4],color=COL['cpl'],alpha=.14,lw=0);ax.fill_between(z,q[:,1],q[:,3],color=COL['cpl'],alpha=.3,lw=0)
 ax.plot(z,q[:,2],color=COL['cpl']);ax.axhline(-1,color='.3',lw=.8,ls='--');ax.set_xlabel(r'Redshift $z$');ax.set_ylabel(r'$w(z)$');ax.text(.96,.95,'CPL: 68 / 95 per cent',ha='right',va='top',transform=ax.transAxes,fontsize=7)
 for ax in axs:ax.tick_params(direction='in',top=True,right=True)
 save(fig,'distance_tests')

def chains():
 fig,axs=plt.subplots(3,3,figsize=(7.1,5.0),layout='constrained');colors=['#0072B2','#D55E00','#009E73','#CC79A7']
 for r,m in enumerate(COL):
  f=np.load(ROOT/f'results/{m}_dovekie_ap_trace.npz');key={'lcdm':'Omega_m','wcdm':'wp','cpl':'wa'}[m]
  for c in range(4):
   axs[r,0].plot(f['draw'],f['H0'][c],lw=.4,alpha=.55,color=colors[c])
   axs[r,1].plot(f['draw'],f[key][c],lw=.4,alpha=.55,color=colors[c])
  axs[r,0].set_ylabel(LAB[m]+'\n'+r'$H_0$');axs[r,1].set_ylabel({'Omega_m':r'$\Omega_m$','wp':r'$w$','wa':r'$w_a$'}[key])
  z=np.load(ROOT/f'chains/{m}_dovekie_ap_baseline_weighted.npz');lw=z['logtarget']-z['logproposal'];lw-=lw.mean()
  for c in range(4):
   ix=z['chain']==c;axs[r,2].plot(z['draw'][ix],lw[ix],'.',ms=2.4,alpha=.7,color=colors[c])
  axs[r,2].axhline(0,color='.4',lw=.4);axs[r,2].set_ylabel(r'$\ln u-\langle\ln u\rangle$')
  for ax in axs[r]:ax.tick_params(direction='in',top=True,right=True)
 for j,t in enumerate(['Expansion rate','Matter / dark energy','Direct-CAMB correction']):axs[0,j].set_title(t,fontsize=9)
 for ax in axs[-1]:ax.set_xlabel('Production step')
 save(fig,'chains')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cmb-only',action='store_true');a=p.parse_args()
 L=Likelihood();cmb(L)
 if not a.cmb_only:posterior();distances(L);chains()
