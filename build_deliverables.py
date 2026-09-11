"""Export identical source into a standalone runner and a readable Colab notebook."""
from pathlib import Path
import json,nbformat,hashlib
sources={str(p):p.read_text() for p in sorted(Path('constellation').glob('*.py'))}
for name in ('run.py','audit.py','calibrate.py','geometry_oracle.py','requirements.txt'):
    sources[name]=Path(name).read_text()
standalone='''#!/usr/bin/env python3
"""Self-contained source export. Install requirements, then use --help."""
import tempfile, pathlib, sys, runpy
SOURCES = '''+repr(sources)+'''
with tempfile.TemporaryDirectory(prefix='constellation-') as directory:
    root=pathlib.Path(directory)
    for name,source in SOURCES.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    sys.path.insert(0,str(root))
    runpy.run_path(str(root/'run.py'),run_name='__main__')
'''
Path('constellation_inference.py').write_text(standalone)
nb=nbformat.v4.new_notebook();cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):cells.append(nbformat.v4.new_code_cell(s))
md('# Constellation detection: classical milestone\nRunnable classical pipeline and diagnostics based only on supplied images and references. This is a measured implementation milestone, not completion of every experiment in PLAN.md. The learned/HPC track remains separate.')
md('## 1. Setup and input\nSelect `smoke`, `evaluate`, or `submission`. Supply a Drive directory or upload a dataset ZIP. An enclosing `participant/` folder is supported. No GPU is required.')
code("import sys, subprocess, os\nif os.environ.get('CONSTELLATION_SKIP_INSTALL') != '1':\n    subprocess.check_call([sys.executable,'-m','pip','install','numpy>=2.0','scipy>=1.14','opencv-python-headless>=4.10','Pillow>=10'])")
code("from pathlib import Path\nimport tempfile, zipfile, json\nMODE = os.environ.get('CONSTELLATION_MODE','evaluate')\nDATA = Path(os.environ.get('CONSTELLATION_DATA','/content/participant'))\nif not DATA.exists():\n    from google.colab import files\n    uploads=files.upload()\n    archive=next(Path(n) for n in uploads if n.endswith('.zip'))\n    DATA=Path('/content/constellation_data');DATA.mkdir(exist_ok=True)\n    with zipfile.ZipFile(archive) as z:\n        for member in z.infolist():\n            if not (DATA/member.filename).resolve().is_relative_to(DATA.resolve()): raise ValueError('Unsafe ZIP path')\n        z.extractall(DATA)\nif not (DATA/'patterns').exists() and (DATA/'participant').exists(): DATA=DATA/'participant'\nassert (DATA/'patterns').exists(), 'Set DATA to the folder containing patterns/train/validation'\nDATA=DATA.resolve()\nprint('Data:',DATA,'Mode:',MODE)")
code("SOURCE = "+repr(sources)+"\nWORK=Path(tempfile.mkdtemp(prefix='constellation-notebook-'))\nfor name,source in SOURCE.items():\n    p=WORK/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\nsys.path.insert(0,str(WORK))\nOUTPUT=Path(os.environ.get('CONSTELLATION_OUTPUT',str(DATA/'outputs/notebook')));OUTPUT.mkdir(parents=True,exist_ok=True)\nprint('Implementation exported to',WORK)")
md('## 2. Inventory and data contracts\nDecode every supplied PNG, record dimensions/channels and decoded hashes, and count labelled categories. Ground truth is used only in evaluation and diagnostics.')
code("subprocess.check_call([sys.executable,str(WORK/'audit.py')],cwd=DATA)\nprint((DATA/'outputs/audit/summary.json').read_text())")
md('## 3. Sky and patch statistics\nThe training counts and inventory above are measured. Absent patches are visually degraded in the same manner as present patches; they cannot be labelled from appearance alone.')
code("import numpy as np\nfrom PIL import Image\nrows=[]\nfor split in ['train','validation']:\n    for p in sorted((DATA/split).glob('*/*_image.png')):\n        a=np.array(Image.open(p));rows.append({'scene':p.parent.name,'mean':float(a.mean()),'std':float(a.std()),'min':int(a.min()),'max':int(a.max())})\nprint(json.dumps(rows,indent=2))")
md('## 4. Reference extraction\nWhite disks in the supplied RGBA diagrams encode stars; green lines encode edges. Extraction uses opaque white connected components, not scene names or manually labelled validation answers. Pair-only references are explicitly ambiguous.')
code("from constellation.references import extract_patterns\npatterns=extract_patterns(DATA/'patterns')\nprint({name:len(points) for name,points in patterns.items()})")
md('## 5. Candidate coverage and duplicate handling\nThe matcher unions dense harmonic retrieval with an optional exhaustive full-image rotation/scale search. Every query retains an independent output. Geometry uses anchor-based grouping within three pixels; image-supported close-source fitting remains an outstanding plan item.')
md('## 6. Classical localization and geometry\n`a0`: translation baseline. `radial`: radial retrieval. `harmonic`: circular harmonic retrieval. `hybrid`: harmonic plus exhaustive search. `final`: hybrid plus ECC and competing geometric fits. All advanced modes verify local rotation/scale candidates. The final pipeline adds ECC alignment and affine quadruple hashing, compares coarse/refined point sets, and uses bounded auxiliary image-star support.')
code("PIPELINE='final'\nTHRESHOLD=float(os.environ.get('CONSTELLATION_THRESHOLD','0.72'))\nsubprocess.check_call([sys.executable,str(WORK/'run.py'),'--data',str(DATA),'--mode',MODE,'--pipeline',PIPELINE,'--threshold',str(THRESHOLD),'--output',str(OUTPUT)])")
md('## 7. Evaluation and limitations\nPublished score: 25% presence, 20% localization, 25% greedy recovery, 30% name. Empty-denominator conventions are unofficial. The three labelled scenes informed method development, so threshold holdouts do not constitute untouched method-selection validation. Geometry oracles must not be described as blind performance.')
code("for name in ['metrics.json','manifest.json']:\n    p=OUTPUT/name\n    if p.exists():print(name,p.read_text())")
md('## 8. Submission and standalone export\nThe script below uses the identical source as this notebook. Submission mode preserves the sample CSV scene set, column order, query count and padding. Verify results before uploading. Smoke mode intentionally does not emit a submission.')
code("STANDALONE = "+repr(standalone)+"\nexport=OUTPUT/'constellation_inference.py';export.write_text(STANDALONE)\nprint('Standalone script:',export)\nif (OUTPUT/'submission.csv').exists():print('CSV:',OUTPUT/'submission.csv')")
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
nbformat.write(nb,'Constellation_Classical.ipynb')
print('Exported standalone and notebook',hashlib.sha256(standalone.encode()).hexdigest())
