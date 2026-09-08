"""Package the completed calculation and record checksums for every archive member."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    for model in ['lcdm','wcdm','cpl']:
        info=json.loads((ROOT/f'results/{model}_dovekie_ap_corrected.json').read_text())['baseline']
        assert info['n_exact']==800 and info['adequate_for_report']
    assert (ROOT/'paper/cosmology_mnras_paper.pdf').is_file()
    assert (ROOT/'results/weighted_diagnostics.json').is_file()
    out=ROOT/'deliverables';out.mkdir(exist_ok=True)
    files=[]
    for folder in ['src','data','chains','results','paper','vendor','references']:
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            rel=p.relative_to(ROOT)
            if '__pycache__' in rel.parts or 'draft' in rel.parts or 'rendered' in rel.parts:continue
            if p.suffix in ['.pyc','.aux','.log','.out','.fls','.fdb_latexmk','.blg','.error']:continue
            if p.name=='library_identity.json':continue
            files.append(p)
    files.extend([ROOT/'README.md',ROOT/'requirements.txt'])
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(out/'cosmology_analysis.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'cosmology_2026/'+str(p.relative_to(ROOT)))
        z.writestr('cosmology_2026/SHA256SUMS.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(out/'cosmology_mnras_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.parent==ROOT/'paper' and p.suffix in ['.tex','.bib','.bst','.cls','.pdf']:z.write(p,p.name)
    for p in out.glob('*.zip'):
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        print(p,p.stat().st_size)

if __name__=='__main__':main()
