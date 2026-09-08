"""Audit the saved numerical outputs without rerunning the Boltzmann solver.

This checks file integrity, selection alignment, weight arithmetic and reporting
gates. It does not constitute an independent scientific replication.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    models = {}
    weighted_diag = json.loads((ROOT / 'results/weighted_diagnostics.json').read_text())
    for model in ['lcdm', 'wcdm', 'cpl']:
        prefix = f'{model}_dovekie_ap'
        summary = json.loads((ROOT / f'results/{prefix}_corrected.json').read_text())
        diag = json.loads((ROOT / f'results/{prefix}_sampling.json').read_text())
        files = sorted((ROOT / 'results/exact').glob(prefix + '_*.npz'))
        assert len(files) == 800
        assert [p.stem.rsplit('_', 1)[-1] for p in files] == [f'{i:05d}' for i in range(800)]
        with np.load(ROOT / f'chains/{prefix}.npz') as z:
            samples, logp = z['samples'], z['logp']
        assert samples.shape[:2] == (4, 30000)
        chain_counts = np.zeros(4, dtype=int)
        for path in files:
            with np.load(path) as z:
                c, d = int(z['chain']), int(z['draw'])
                assert np.array_equal(z['v'], samples[c, d])
                assert float(z['logproposal']) == float(logp[c, d])
                assert np.isfinite(z['y']).all()
                assert np.isclose(float(z['logtarget']), -.5 * z['components'].sum(), rtol=0, atol=1e-10)
                chain_counts[c] += 1
        assert np.array_equal(chain_counts, [200] * 4)
        assert diag['max_rhat'] < 1.01
        assert min(diag['min_ess_bulk'], diag['min_ess_tail']) > 400
        variants = {}
        for variant, q in summary.items():
            with np.load(ROOT / f'chains/{prefix}_{variant}_weighted.npz') as z:
                lw = z['logtarget'] - z['logproposal']
                w = np.exp(lw - lw.max()); w /= w.sum()
                assert len(w) == 800 and np.isfinite(w).all()
                assert np.allclose(w, z['weights'], rtol=1e-12, atol=1e-15)
                assert np.isclose(1 / (w @ w), q['weight_ess'])
                mean = w @ z['parameters']
                for i, name in enumerate(z['names']):
                    assert np.isclose(mean[i], q['parameters'][name]['mean'])
            ok = q['weight_ess'] >= 400 and q['pareto_k'] < .7
            if variant == 'baseline':
                assert ok and q['n_exact'] == 800
            variants[variant] = {
                'weight_ess': q['weight_ess'], 'pareto_k': q['pareto_k'],
                'parameter_constraints_reportable': bool(ok),
                'max_mcse_over_posterior_sd': weighted_diag[model][variant]['max_mcse_over_posterior_sd'],
            }
        models[model] = {
            'direct_draws': len(files), 'direct_draws_per_chain': chain_counts.tolist(),
            'production_draws': int(samples.shape[0] * samples.shape[1]),
            'max_rhat': diag['max_rhat'], 'min_bulk_ess': diag['min_ess_bulk'],
            'min_tail_ess': diag['min_ess_tail'], 'variants': variants,
        }
    for item in json.loads((ROOT / 'data/manifest.json').read_text()):
        assert 'sha256' in item, item
        assert hashlib.sha256((ROOT / item['file']).read_bytes()).hexdigest() == item['sha256'], item['file']
    out = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'numerical_completion_gates_passed': True,
        'independent_scientific_review': 'not performed',
        'sensitivity_qualification': 'No-supernova wCDM and CPL constraints are excluded because importance ESS fails the reporting threshold. No alternative CMB or lensing refits were performed.',
        'models': models,
        'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'matplotlib', 'camb', 'cobaya', 'sacc', 'astropy', 'arviz']},
    }
    (ROOT / 'results/completion_evidence.json').write_text(json.dumps(out, indent=2))
    print('All saved numerical completion checks passed; 2400 direct draws and 360000 production steps verified.')


if __name__ == '__main__':
    main()
