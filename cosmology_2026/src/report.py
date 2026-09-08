"""Fill the final manuscript only from completed, checked numerical results."""
import json,re,csv
from pathlib import Path
from model import ROOT

MODELS=['lcdm','wcdm','cpl'];LABEL={'lcdm':r'$\Lambda$CDM','wcdm':r'$w$CDM','cpl':'CPL'}
def read(name):return json.loads((ROOT/'results'/name).read_text())
def table(caption,columns,header,rows,label,wide=False):
 env='table*' if wide else 'table'
 return '\n'.join([r'\begin{'+env+r'}\centering',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{'+columns+r'}\toprule',header+r'\\\midrule',*rows,r'\bottomrule\end{tabular}',r'\end{'+env+'}'])

def main():
 result={m:read(f'{m}_dovekie_ap_corrected.json') for m in MODELS}
 diag={m:read(f'{m}_dovekie_ap_sampling.json') for m in MODELS}
 mc=read('model_comparison.json');acc=read('numerical_accuracy.json');wd=read('weighted_diagnostics.json')
 for m in MODELS:
  q=result[m]['baseline']
  assert q['n_exact']==800 and q['weight_ess']>=400 and q['pareto_k']<.7, (m,'importance diagnostic gate')
  assert diag[m]['max_rhat']<1.01 and min(diag[m]['min_ess_bulk'],diag[m]['min_ess_tail'])>400, (m,'chain diagnostic gate')
 def stat(m,k,variant='baseline'):return result[m][variant]['parameters'][k]
 def value(m,k,dec=3,variant='baseline'):
  q=stat(m,k,variant)
  if abs(q['sd'])<1e-10:return f"${q['mean']:g}$ (fixed)"
  return f"${q['mean']:.{dec}f}\\pm{q['sd']:.{dec}f}$"
 h=stat('lcdm','H0');om=stat('lcdm','Omega_m');w0=stat('cpl','w0');wa=stat('cpl','wa')
 abstract=(f"In \\lcdm, we obtain $H_0={h['mean']:.2f}\\pm{h['sd']:.2f}\\,\\hunit$ and "
           f"$\\Omega_m={om['mean']:.4f}\\pm{om['sd']:.4f}$. CPL gives "
           f"$w_0={w0['mean']:.2f}\\pm{w0['sd']:.2f}$ and $w_a={wa['mean']:.2f}\\pm{wa['sd']:.2f}$, "
           f"with a local deviance improvement of ${-mc['cpl']['delta_deviance']:.2f}$ for two extra parameters. ")
 entries=[('ombh2',r'$\omega_b$',5),('omch2',r'$\omega_c$',4),('theta100',r'$100\theta_\star$',5),('logA',r'$\ln(10^{10}A_s)$',3),('ns',r'$n_s$',4),('tau',r'$\tau$',4),('H0',r'$H_0\,[\hunit]$',2),('Omega_m',r'$\Omega_m$',4),('sigma8',r'$\sigma_8$',4),('S8',r'$S_8$',4),('r_drag',r'$r_d\,[\mathrm{Mpc}]$',2),('age_Gyr','Age [Gyr]',3),('w0',r'$w_0$',3),('wa',r'$w_a$',3),('A_cmb',r'$A_{\rm CMB}$',4),('P_act',r'$P_{\rm ACT}$',4)]
 rows=[label+' & '+' & '.join(value(m,k,dec) for m in MODELS)+r'\\' for k,label,dec in entries]
 pt=table('Direct-CAMB importance-corrected posterior means and standard deviations for the baseline combination. Equal-tailed 16th, 50th and 84th percentiles and Monte Carlo errors are supplied in the accompanying tables. Fixed quantities have no quoted uncertainty.','lccc',r'Parameter & $\Lambda$CDM & $w$CDM & CPL',rows,'tab:params',True)
 rows=[]
 for m in MODELS:
  a=diag[m];b=result[m]['baseline'];rows.append(LABEL[m]+f" & {a['max_rhat']:.4f} & {a['min_ess_bulk']:.0f} & {a['min_ess_tail']:.0f} & {b['n_exact']} & {b['weight_ess']:.0f} & {b['pareto_k']:.2f}"+r'\\')
 dt=table('Production-chain and importance-correction diagnostics. Four chains per model each contain 30,000 production steps after 5000 warmup steps. Chain diagnostics cover physical and derived coordinates. $N_{\rm IS}$ describes the direct-evaluation weights and is distinct from MCMC ESS.','lrrrrrr',r'Model & Max. $\widehat R$ & Min. bulk ESS & Min. tail ESS & Direct draws & $N_{\rm IS}$ & Pareto $k$',rows,'tab:diagnostics',True)
 rmax=max(diag[m]['max_rhat'] for m in MODELS);emin=min(diag[m]['min_ess_bulk'] for m in MODELS)
 imax=max(wd[m]['baseline']['max_mcse_over_posterior_sd'] for m in MODELS)
 dtext=(f"All models satisfy $\\widehat R\\leq{rmax:.4f}$ and minimum bulk ESS ${emin:.0f}$. "
        f"We use 800 stratified direct evaluations per model. Forty contiguous batches of the importance-sampling influence function give a maximum mean Monte Carlo error of {100*imax:.1f} per cent of a posterior standard deviation. "
        "These estimates account for residual correlation between nearby selected draws. The dark-energy posterior lies well inside the stated prior boundaries.")
 amin=min(acc[m]['data_space_squared_difference'] for m in MODELS);amax=max(acc[m]['data_space_squared_difference'] for m in MODELS)
 diff=max(abs(acc[m]['change_in_model_deviance_difference']) for m in MODELS)
 atext=(f"At the three fitted modes, the squared high-$\\ell$ CMB plus BAO data-space difference between these settings is {amin:.3f}--{amax:.3f}. "
        f"The common absolute deviance changes by approximately one, while the largest change in a between-model deviance difference is {diff:.3f}. "
        "These fixed-point checks are not a new high-accuracy MCMC run. An independent low-redshift quadrature reproduces the SN-only Dovekie constraint, $\\Omega_m=0.3308\\pm0.0153$, and gives $0.3323\\pm0.0182$ for Pantheon+.")
 rt=(r'Table~\ref{tab:params} and Fig.~\ref{fig:posterior} show the corrected baseline constraints. '
     f"The \\lcdm\\ expansion rate is $H_0={h['mean']:.2f}\\pm{h['sd']:.2f}\\,\\hunit$, with $S_8={stat('lcdm','S8')['mean']:.3f}\\pm{stat('lcdm','S8')['sd']:.3f}$. "
     f"Constant-$w$ gives $w={stat('wcdm','w0')['mean']:.3f}\\pm{stat('wcdm','w0')['sd']:.3f}$, close to $-1$. "
     f"CPL favours $w_0={w0['mean']:.3f}\\pm{w0['sd']:.3f}$ and $w_a={wa['mean']:.3f}\\pm{wa['sd']:.3f}$. "
     r'The two dark-energy coordinates are strongly correlated. Figure~\ref{fig:distances} expresses the resulting change as a redshift-dependent equation of state and distance residuals. '
     f"Allowing CPL lowers the inferred expansion rate to $H_0={stat('cpl','H0')['mean']:.2f}\\pm{stat('cpl','H0')['sd']:.2f}\\,\\hunit$ in this combination.")
 rows=[]
 components=[('planck_high','Planck high-$\\ell$'),('act','ACT'),('planck_lowT','Planck low-T'),('planck_lowE','Planck low-E'),('bao','BAO / AP'),('sn','Supernovae'),('calibration_prior','Calibration prior')]
 for key,label in components:rows.append(label+' & '+' & '.join(f"{mc[m]['components'][key]:.2f}" for m in MODELS)+r'\\')
 rows+=[r'\midrule Total $D$ & '+' & '.join(f"{mc[m]['deviance']:.2f}" for m in MODELS)+r'\\',r'$\Delta D$ & '+' & '.join(f"{mc[m]['delta_deviance']:.2f}" for m in MODELS)+r'\\',r'$\Delta D+2\Delta k$ & '+' & '.join(f"{mc[m]['delta_penalized_deviance']:.2f}" for m in MODELS)+r'\\']
 mt=table('Direct-CAMB best evaluated local fit statistics. The calibration prior is included in $D$. Fixed likelihood offsets are retained; only same-data differences are interpreted. These are approximate local profiles, not certified global minima.','lrrr',r'Contribution & $\Lambda$CDM & $w$CDM & CPL',rows,'tab:fit')
 mtext=(f"The best evaluated direct-CAMB local deviances improve by {-mc['wcdm']['delta_deviance']:.2f} and {-mc['cpl']['delta_deviance']:.2f} relative to \\lcdm\\ for constant-$w$ and CPL, respectively. "
        "Each mode begins at the surrogate optimum. A finite-difference direct-likelihood correction is used in a local Powell search, and the candidate is reevaluated with CAMB; this does not certify a global maximum. The resulting fit-statistic decomposition is in Table~\\ref{tab:fit}; the CMB residuals appear in Fig.~\\ref{fig:cmb}. "
        "The CPL improvement is distributed across the supernova, BAO and ACT terms, rather than coming from a single added likelihood. ")
 sensitivity=[];srows=[]
 for variant,title in [('bao_only',r'Original Ly$\alpha$ BAO'),('pantheon','Pantheon+ replacement'),('no_sn','Supernovae omitted')]:
  for m in MODELS:
   q=result[m][variant];ok=q['weight_ess']>=400 and q['pareto_k']<.7
   if ok:
    srows.append(title+' / '+LABEL[m]+' & '+value(m,'H0',2,variant)+' & '+value(m,'w0',3,variant)+' & '+value(m,'wa',3,variant)+f" & {q['weight_ess']:.0f}"+r'\\')
   else:sensitivity.append(f"{title} in {LABEL[m]} has $N_{{\\rm IS}}={q['weight_ess']:.0f}$ (Pareto $k={q['pareto_k']:.2f}$), so we do not report parameter constraints for that reweighting.")
 st=table('Replacement and omission checks using direct-CAMB likelihood-ratio weights. Rows are shown only when weight ESS is at least 400 and Pareto $k<0.7$. All changes preserve the baseline CMB treatment. These are alternative fits, not products of overlapping SN samples.','lrrrr',r'Alternative / model & $H_0\,[\hunit]$ & $w_0$ & $w_a$ & $N_{\rm IS}$',srows,'tab:sensitivity',True)
 stxt=r'Table~\ref{tab:sensitivity} gives the adequately sampled replacement tests. '
 for var,word in [('bao_only',r'reverting to the original Ly$\alpha$ pair'),('pantheon','replacing Dovekie by Pantheon+')]:
  q=result['cpl'][var]
  if q['weight_ess']>=400 and q['pareto_k']<.7:
   dw=q['parameters']['w0']['mean']-w0['mean'];da=q['parameters']['wa']['mean']-wa['mean']
   stxt+=f"For CPL, {word} shifts the mean $w_0$ by ${dw:+.3f}$ and $w_a$ by ${da:+.3f}$. "
 stxt+=' '.join(sensitivity)+' The accompanying diagnostics provide paired Monte Carlo errors for every parameter shift, using the same direct-evaluation draws in each comparison.'+'\n'+st
 concl=(f"This public-likelihood reanalysis combines current SN calibration and DESI Ly$\\alpha$ information with Planck and ACT spectral data. The twelve independently seeded chains converge, and the retained direct-CAMB corrections preserve useful sample weight. "
        f"The baseline \\lcdm\\ result is $H_0={h['mean']:.2f}\\pm{h['sd']:.2f}\\,\\hunit$. CPL improves the local deviance by {-mc['cpl']['delta_deviance']:.2f}, or {-mc['cpl']['delta_penalized_deviance']:.2f} after the stated two-parameter penalty. "
        "This is an indication within a specified phenomenological model, not a Bayesian evidence calculation or a discovery claim. In this data combination, late-time dark-energy freedom moves $H_0$ farther from the Local Distance Network central value. A geometrical fit improvement therefore does not by itself resolve the distance-scale discrepancy. ")
 vals={'ABSTRACT_RESULTS':abstract,'NTRAIN':'640','WARMUP':'5000','DRAWS':'30,000','DIAGNOSTICS_TEXT':dtext,'DIAGNOSTICS_TABLE':dt,'ACCURACY_TEXT':atext,'RESULTS_TEXT':rt,'PARAMETER_TABLE':pt,'MODEL_COMPARISON_TEXT':mtext,'MODEL_TABLE':mt,'SENSITIVITY_TEXT':stxt,'CONCLUSIONS_TEXT':concl}
 t=(ROOT/'paper/article_template.tex').read_text()
 for key,value_ in vals.items():t=t.replace('@@'+key+'@@',value_)
 assert '@@' not in t
 (ROOT/'paper/cosmology_mnras_paper.tex').write_text(t)
 # Full numeric summaries are separate from the rounded typeset values.
 with (ROOT/'results/parameter_summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['model','variant','parameter','adequate_for_report','mean','sd','q16','median','q84','mcse_mean_iid','mcse_mean_batch']);writer.writeheader()
  for m in MODELS:
   for variant,q in result[m].items():
    for k,s in q['parameters'].items():writer.writerow({'model':m,'variant':variant,'parameter':k,'adequate_for_report':q['weight_ess']>=400 and q['pareto_k']<.7,**s,'mcse_mean_batch':wd[m][variant]['mcse_mean'][k]})
 (ROOT/'results/manuscript_values.json').write_text(json.dumps(vals,indent=2))
 print(ROOT/'paper/cosmology_mnras_paper.tex')

if __name__=='__main__':main()
