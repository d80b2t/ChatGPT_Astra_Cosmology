"""Create an explicitly provisional reading copy while exact checks continue."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,shutil,re
import numpy as np
import matplotlib.pyplot as plt
from model import ROOT
import figures

OUT=ROOT/'paper/draft';OUT.mkdir(exist_ok=True)
data={}
for m in figures.COL:
 z=np.load(ROOT/f'results/{m}_dovekie_ap_trace.npz')
 data[m]={k:z[k].ravel() for k in z.files if k!='draw'}
 if m=='lcdm':data[m]['w0']=np.full_like(data[m]['H0'],-1.);data[m]['wa']=np.zeros_like(data[m]['H0'])
 if m=='wcdm':data[m]['w0']=data[m]['wp'];data[m]['wa']=np.zeros_like(data[m]['H0'])
def provisional(m,variant='baseline'):
 d={k:a[::3] for k,a in data[m].items()};n=len(d['H0'])
 return d,np.ones(n)/n
def save(fig,name):
 fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.035)
 fig.savefig(OUT/f'{name}.png',dpi=130,bbox_inches='tight',pad_inches=.035);plt.close(fig)
figures.weighted=provisional;figures.save=save;figures.posterior()

def estimate(m,p,dp=3):
 a=data[m][p]
 if np.std(a)<1e-12:return f'${a[0]:g}$ (fixed)'
 return f'${np.mean(a):.{dp}f}\\pm{np.std(a):.{dp}f}$'
rows=[]
for p,label,dp in [('ombh2',r'$\omega_b$',5),('omch2',r'$\omega_c$',4),('theta100',r'$100\theta_\star$',5),('logA',r'$\ln(10^{10}A_s)$',3),('ns',r'$n_s$',4),('tau',r'$\tau$',4),('H0',r'$H_0\,[\hunit]$',2),('Omega_m',r'$\Omega_m$',4),('sigma8',r'$\sigma_8$',4),('S8',r'$S_8$',4),('r_drag',r'$r_d\,[\mathrm{Mpc}]$',2),('age_Gyr',r'Age [Gyr]',3),('w0',r'$w_0$',3),('wa',r'$w_a$',3)]:
 rows.append(label+' & '+' & '.join(estimate(m,p,dp) for m in figures.COL)+r'\\')
table=r'''\begin{table*}\centering
\caption{Provisional surrogate-posterior means and standard deviations. The chains are complete; direct-CAMB correction of the final posterior tables is still in progress. Rounded summaries use uniformly thinned stored production draws. Fixed quantities have no quoted uncertainty.}\label{tab:params}
\begin{tabular}{lccc}\toprule Parameter & $\Lambda$CDM & $w$CDM & CPL\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table*}'
diag=[]
for m in figures.COL:
 j=json.loads((ROOT/f'results/{m}_dovekie_ap_sampling.json').read_text())
 diag.append(figures.LAB[m]+f" & {j['max_rhat']:.4f} & {j['min_ess_bulk']:.0f} & {j['min_ess_tail']:.0f}"+r'\\')
dt=r'''\begin{table}\centering\caption{Completed production-chain diagnostics, including derived parameters. Each of four chains has 30,000 production steps after 5,000 discarded warmup steps. These diagnose the surrogate posterior.}\begin{tabular}{lrrr}\toprule Model & Max. $\widehat R$ & Min. bulk ESS & Min. tail ESS\\\midrule
'''+ '\n'.join(diag)+r'\bottomrule\end{tabular}\end{table}'
mc=json.loads((ROOT/'results/model_comparison.json').read_text());mr=[]
for m in figures.COL:
 j=mc[m];mr.append(figures.LAB[m]+f" & {j['deviance']:.2f} & {j['delta_deviance']:.2f} & {j['delta_penalized_deviance']:.2f}"+r'\\')
mt=r'''\begin{table}\centering\caption{Direct-CAMB best evaluated local deviances, with the calibration prior included. These are approximate local profiles; final numerical-accuracy checks remain pending.}\begin{tabular}{lrrr}\toprule Model & $D$ & $\Delta D$ & $\Delta D+2\Delta k$\\\midrule
'''+ '\n'.join(mr)+r'\bottomrule\end{tabular}\end{table}'
status=r'''\noindent\fbox{\parbox{0.94\columnwidth}{\textbf{Provisional reading copy.} Twelve MCMC chains are complete. The parameter table and contours currently describe the fast numerical likelihood. Direct-CAMB importance correction, data-replacement tests and final numerical checks are still in progress. This draft must not be cited as a completed analysis.}}\medskip
'''
tokens={
 'ABSTRACT_RESULTS':r'The provisional $\Lambda$CDM posterior gives $H_0\simeq68.26\pm0.27\,\hunit$ and $\Omega_m\simeq0.303\pm0.004$. Direct evaluations near the fitted modes favour CPL by $\Delta D\simeq-10.44$ before the two-parameter penalty. These results remain provisional pending completion of the validation and replacement tests.',
 'NTRAIN':'640','WARMUP':'5000','DRAWS':'30,000',
 'DIAGNOSTICS_TEXT':r'The completed chains satisfy $\widehat R<1.003$, with minimum bulk ESS above 3000 across the fitted and derived coordinates. Direct-likelihood corrections are being calculated for 800 selected draws per model. Correction ESS, tail diagnostics and Monte Carlo uncertainty will be added after those runs finish.',
 'DIAGNOSTICS_TABLE':dt,
 'ACCURACY_TEXT':r'At the reference cosmology, the squared high-$\ell$ CMB plus BAO data-space difference between these settings is 0.0294. Checks around the fitted modes are pending. A separate, non-CAMB, low-redshift quadrature reproduces the Dovekie SN-only constraint, $\Omega_m=0.3308\pm0.0153$, and gives $0.3323\pm0.0182$ for Pantheon+.',
 'RESULTS_TEXT':r'Table~\ref{tab:params} and Fig.~\ref{fig:posterior} show the completed surrogate-chain results. These numerical estimates are a reading-copy preview; final means, intervals and contours will use direct-CAMB importance corrections. The early corrected $\Lambda$CDM draws show small log-likelihood differences, but this does not establish the accuracy of the other models.',
 'PARAMETER_TABLE':table,
 'MODEL_COMPARISON_TEXT':r'Direct-CAMB calculations near each surrogate mode give the local fit statistics below. The CPL improvement relative to $\Lambda$CDM is about 10.44 in deviance, while constant-$w$ improves it by about 1.06. These evaluations use the actual observational likelihoods; the optima have received a local gradient-based correction and are not certified global minima. The CPL improvement is distributed across the SN, BAO and ACT terms. Its penalized-deviance advantage is approximately 6.44.',
 'MODEL_TABLE':mt,
 'SENSITIVITY_TEXT':r'\textbf{Results pending.} Pantheon+ replaces Dovekie, and the original Ly$\alpha$ BAO pair replaces the updated AP+BAO pair, in separate likelihood-ratio reweighting tests. Their parameter shifts and effective sample sizes will be reported only after the direct evaluations finish. No numerical robustness claim is made in this reading copy.',
 'CONCLUSIONS_TEXT':r'The completed MCMC runs establish a workable joint spectral-CMB, BAO and SN analysis with good chain diagnostics. Preliminary constraints remain near the standard inverse-distance-ladder expansion rate, while the evolving-dark-energy model improves the local joint fit. The size and robustness of that preference require the outstanding direct-likelihood and data-choice checks. The final conclusions will be written after those checks, rather than inferred from convergence of an approximate likelihood alone.'}
t=(ROOT/'paper/article_template.tex').read_text()
for k,v in tokens.items():t=t.replace('@@'+k+'@@',v)
t=t.replace(r'\section{Introduction}',r'\section{Introduction}'+'\n'+status)
t=t.replace('retained draws are recalculated with CAMB and importance-corrected to the direct\nlikelihood. We report both chain convergence and correction efficiency.','retained draws are currently being recalculated with CAMB for importance correction\nto the direct likelihood. We report completed chain diagnostics; correction-efficiency\nand data-sensitivity results remain pending.')
t=t.replace('Pantheon+ and the previous Ly$\\alpha$ BAO measurement provide sensitivity tests.','Pantheon+ and the previous Ly$\\alpha$ BAO measurement are planned replacement tests in the running pipeline.')
t=t.replace('The final parameter tables and credible regions use these\ndirect-likelihood weights.','The completed analysis will use these direct-likelihood weights; the current\nreading-copy table and contours still describe the surrogate posterior.')
t=t.replace('Direct-CAMB importance-corrected posterior constraints.','Provisional surrogate-posterior constraints; direct-CAMB correction is pending.')
t=re.sub(r'\\begin\{figure\*\}\s*\\centering\\includegraphics\[width=\\textwidth\]\{distance_tests.pdf\}.*?\\end\{figure\*\}','',t,flags=re.S)
t=t[:t.index(r'\appendix')]+r'\label{lastpage}'+'\n'+r'\end{document}'+'\n'
t=t.replace('The accompanying archive contains','The final accompanying archive will contain').replace('The attached data, code, weighted draws and diagnostics make the present\ncalculation inspectable and reproducible.','The final data, code, weighted draws and diagnostics will make the calculation\ninspectable and reproducible.')
t=t.replace('The analysis was executed on 4--5 September 2026.','The analysis was started on 4 September 2026 and remains in progress on 5 September.')
assert '@@' not in t
(OUT/'cosmology_mnras_draft.tex').write_text(t)
for f in ['mnras.cls','mnras.bst','references.bib']:shutil.copy2(ROOT/'paper'/f,OUT/f)
print(OUT/'cosmology_mnras_draft.tex',flush=True)
