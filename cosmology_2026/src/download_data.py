"""Download immutable public inputs, recording provenance and SHA-256 hashes."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request, zipfile, tarfile, io

ROOT = Path(__file__).resolve().parents[1]
RAW = 'https://raw.githubusercontent.com/'
MANIFEST = ROOT/'data/manifest.json'
EXPECTED = {r['file']:r for r in json.loads(MANIFEST.read_text()) if 'sha256' in r} if MANIFEST.exists() else {}
DES = RAW+'des-science/DES-SN5YR/c9a4fcafc4cbd19bd750dee47fc76194a45c181f/'
BAO = RAW+'CobayaSampler/bao_data/bb0c1c9009dc76d1391300e169e8df38fd1096db/'
ACT = RAW+'ACTCollaboration/DR6-ACT-lite/627aeafb88ae5ad1aa66b406bea2d65cfa66a27d/'
PP = RAW+'PantheonPlusSH0ES/DataRelease/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon+_Data/4_DISTANCES_AND_COVAR/'
FILES = {
 'data/desi_dr2_mean.txt':BAO+'desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt',
 'data/desi_dr2_cov.txt':BAO+'desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt',
 'data/dovekie.csv':DES+'4_DISTANCES_COVMAT/DES-Dovekie_HD.csv',
 'data/dovekie_precision.npz':DES+'4_DISTANCES_COVMAT/STAT+SYS.npz',
 'data/dovekie_readme.md':DES+'4_DISTANCES_COVMAT/README.md',
 'data/pantheon_plus.dat':PP+'Pantheon+SH0ES.dat',
 'data/pantheon_plus.cov':PP+'Pantheon+SH0ES_STAT+SYS.cov',
 'vendor/dovekie_likelihood.py':DES+'4_DISTANCES_COVMAT/DES-Dovekie-SN_Likelihood.py',
 'vendor/act_dr6_cmbonly.py':ACT+'act_dr6_cmbonly/act_dr6_cmbonly.py',
 'vendor/PlanckActCut.py':ACT+'act_dr6_cmbonly/PlanckActCut.py',
 'vendor/PlanckActCut.yaml':ACT+'act_dr6_cmbonly/PlanckActCut.yaml',
 'data/dr6_data_cmbonly.tar.gz':'https://lambda.gsfc.nasa.gov/data/act/pspipe/sacc_files/dr6_data_cmbonly.tar.gz',
 'data/plik_lite_2018.zip':'https://github.com/CobayaSampler/planck_native_data/releases/download/v1/plik_lite_2018_AL.zip',
 'data/planck_lowT.zip':'https://github.com/CobayaSampler/planck_native_data/releases/download/v1/planck_2018_lowT.zip',
 'data/planck_lowE.zip':'https://github.com/CobayaSampler/planck_native_data/releases/download/v1/planck_2018_lowE.zip',
 'paper/mnras.cls':'https://mirrors.ctan.org/macros/latex/contrib/mnras/mnras.cls',
 'paper/mnras.bst':'https://mirrors.ctan.org/macros/latex/contrib/mnras/mnras.bst',
}

def fetch(kv):
 name,url = kv; path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True)
 try:
  expected=EXPECTED.get(name)
  content=path.read_bytes() if path.exists() else None
  if content is None or (expected and hashlib.sha256(content).hexdigest()!=expected['sha256']):
   req=urllib.request.Request(url,headers={'User-Agent':'Reproducible cosmology research'})
   with urllib.request.urlopen(req,timeout=90) as response: content=response.read()
   if expected and hashlib.sha256(content).hexdigest()!=expected['sha256']:
    raise ValueError('Downloaded bytes differ from pinned SHA-256: '+name)
   temporary=path.with_suffix(path.suffix+'.part');temporary.write_bytes(content);temporary.replace(path)
  if name.endswith('.zip'):
   with zipfile.ZipFile(io.BytesIO(content)) as archive:
    if archive.testzip() is not None:raise ValueError('Corrupt ZIP member: '+name)
  if expected:return expected
  record={'file':name,'url':url,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'retrieved':'2026-09-04'}
  print(name,len(content),flush=True)
  return record
 except Exception as e:
  print('FAILED',name,str(e),flush=True)
  return {'file':name,'url':url,'error':str(e)}

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  records=list(pool.map(fetch,FILES.items()))
 if any('error' in r for r in records):raise RuntimeError('Input recovery failed; original manifest preserved')
 MANIFEST.write_text(json.dumps(records,indent=2))
 for name in ['plik_lite_2018.zip','planck_lowT.zip','planck_lowE.zip']:
  path=ROOT/'data'/name
  if path.exists():
   with zipfile.ZipFile(path) as z:z.extractall(ROOT/'data/cmb')
 path=ROOT/'data/dr6_data_cmbonly.tar.gz'
 if path.exists():
  with tarfile.open(path) as t:t.extractall(ROOT/'data/cmb/ACTDR6CMBonly',filter='data')
