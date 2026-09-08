# Multi-probe cosmology, 7 September 2026

This is a reproducible research draft prepared at Nic Ross's request. It contains
newly executed cosmological parameter inference, not a compilation relabeled as
new chains. The paper distinguishes fitted likelihoods from literature context.
It has not undergone independent scientific review.

## Fitted observations

- Planck 2018 Plik-lite TT, TE, EE with bin upper edges at most 1000, 600, 600,
  respectively; Commander low-T and SimAll low-E at multipoles 2–29.
- ACT DR6 foreground-marginalized TT, TE, EE, effective multipoles 600–6500.
  Full supplied bandpower windows are integrated through multipole 8501.
- DESI DR2's 13-dimensional BAO vector and full covariance.
- The July 2026 DESI Ly-alpha AP+BAO distance pair from equation 26 of
  arXiv:2607.27410v3, replacing the existing Ly-alpha BAO pair. It is not appended
  as an independent duplicate. The quoted DM/rd, DH/rd correlation is retained.
- DES-Dovekie: 1820 supernovae, using the published total precision matrix.
  The .csv filename contains SNANA whitespace text in the pinned release.
- Pantheon+: an alternative supernova likelihood (zHD > 0.01; 1590 light curves),
  with the full statistical-plus-systematic covariance and marginalized offset.
  It is not multiplied by Dovekie. SH0ES calibrator distances are not used.

DES Y6, KiDS-Legacy, SPT-3G, CMB lensing reconstruction, cluster abundance,
distance ladders, gravitational-wave sirens, and other surveys discussed in the
paper are external comparisons unless explicitly identified otherwise.

## Models and assumptions

Flat LambdaCDM, flat constant-w CDM, and flat CPL w0–wa CDM. Six usual physical
CMB parameters are varied, plus the dark-energy parameters as appropriate and
two instrumental calibration parameters. The sound horizon is calculated by
CAMB, not held fixed. Neutrino mass is fixed to 0.06 eV, Neff to 3.046, spatial
curvature to zero, primordial tensors and running to zero. PPF handles dark
energy crossing w=-1. CAMB's BBN-consistent helium and Recfast recombination
are used. These are conditional-model constraints, not neutrino-mass limits.

## Sampling and exact correction

`src/model.py` reads the observations and evaluates direct CAMB predictions.
`src/emulator.py` fits a local polynomial approximation to theory predictions.
`src/sample.py` runs four independent Metropolis chains for each model, adapting
only during discarded warmup, then freezing the transition kernel. These chains
sample the emulator posterior. Rank-normalized folded/split R-hat, bulk/tail
ESS, and acceptance are saved.

`src/correct.py` selects stratified draws from all four chains, recalculates
CAMB spectra and individual-supernova predictions, and records the exact target
and proposal log densities. The final weighted posterior uses
`weight = exp(log_target_direct_CAMB - log_proposal_emulator)` and reports weight
ESS and Pareto-k. The uncorrected emulator chains are explicitly labeled.
The direct target still has the numerical settings and astrophysical
approximations listed in the paper. Weighted ESS is not chain ESS; they are
reported separately.

An interpolation grid accelerates the supernova term only in the sampling
proposal. Exact retained-sample likelihoods evaluate every supernova at its
individual Hubble-diagram and heliocentric redshifts. The common absolute
magnitude/intercept is marginalized with the full covariance. Its fixed
normalization cancels within same-dataset model comparisons.

## Reproducing

Python 3.12, a BLAS implementation, and a CAMB-compatible CPU are required.
Use a fresh virtual environment and install `requirements.txt`. Limit numerical
threading when using several worker processes:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python src/download_data.py
python src/emulator.py --n 640 --workers 8
python src/sample.py lcdm
python src/sample.py wcdm
python src/sample.py cpl
python src/correct.py lcdm --workers 8
python src/correct.py wcdm --workers 8
python src/correct.py cpl --workers 8
```

The model-specific execution counts and seeds in `results/` are authoritative.
Data downloads are pinned to upstream Git commits or release assets; manifests
retain URLs and SHA-256 digests. The manually transcribed AP measurement has a
separate provenance file. Supplied raw observational data retain their original
scientific ownership and citation requirements.

The analysis does not resample published posterior chains. Proposal fitting,
theory-bank creation, MCMC, and exact corrections were run in this workspace.
Exact point files support interrupted-job recovery. Do not reuse cached exact
point filenames for a different MCMC chain or number of selected points.

## Files

- `paper/`: MNRAS class, LaTeX source, figures and compiled manuscript.
- `src/`: executable analysis and reporting code.
- `data/`: release inputs, covariance files and provenance manifests.
- `chains/`: emulator MCMC and direct-CAMB importance-weighted samples.
- `results/`: diagnostics, numerical validation, parameter tables and logs.

The paper's model-comparison statistics are conditional on the stated data,
priors, and theory. An AIC difference is not a Bayes factor. A two-parameter
likelihood-ratio improvement is not a one-parameter Gaussian significance.
Cross-survey literature estimates must not be averaged without accounting for
overlap, nuisance parameters, shared calibration, and model dependence.

## Finishing a checkpointed run

The correction worker pool resumes only matching selected draws and proposal
log densities. For the saved 800-point selection, run:

```bash
python src/correct_all.py lcdm wcdm cpl --n 800 --workers 8
python src/diagnostics.py --weighted
python src/figures.py
python src/report.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error cosmology_mnras_paper.tex
```

The report refuses baseline results unless every model has 800 direct draws,
weight ESS at least 400, Pareto k below 0.7, R-hat below 1.01, and bulk/tail
chain ESS above 400. Sensitivity variants that fail the importance diagnostic
are explicitly qualified and excluded from the manuscript parameter table.
The original chain diagnostics are in the model-specific diagnostics CSV files;
weighted mean MCSE and paired sensitivity-shift MCSE are in
`results/weighted_diagnostics.json`.

The checkpointed runs were resumed on 7 September after repairing the Python
environment and restoring a truncated local Pantheon+ covariance file. The
restored public file was verified against the original SHA-256 manifest.
The Planck low-T download archive was also restored; all five unpacked
likelihood files match the complete release exactly. Recovery details are in
`results/restart_lowT_integrity.json`. The download script now checks pinned
hashes and ZIP integrity, and preserves the manifest if recovery fails.

The stored Planck low-T download archive was also restored in full. Every
unpacked low-T file used in inference matched the complete public release;
the likelihood data did not require replacement. The prior archive hash and
restoration audit are preserved in `results/restart_lowT_integrity.json`.
The downloader now validates archive contents before recording their hashes.

## Regenerating validation and publication outputs

From the repository root, after sampling and corrections:

```bash
python src/diagnostics.py
python src/bestfit.py --workers 8
python src/validate_accuracy.py
python src/validate_likelihoods.py
python src/validate_supernovae.py
python src/diagnostics.py --weighted
python src/figures.py
python src/report.py
python src/verify_outputs.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/cosmology_mnras_paper.tex
```

`verify_outputs.py` audits the saved draw/chain correspondence, normalized
weights, reporting thresholds and public-input hashes. It does not replace
independent scientific replication. `results/completion_evidence.json` records
the completed audit and the package versions used for final reporting.

All three baseline corrections are complete at 800 direct evaluations per
model. The no-supernova wCDM and CPL reweightings fail the weight-ESS threshold;
their files are retained with reportability flags for diagnosis, and their
parameter estimates are excluded from the paper. Separate chains would be
required before using those two variants as quantitative constraints.
