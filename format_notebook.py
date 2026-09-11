from pathlib import Path
import nbformat
nb=nbformat.read('Constellation_Classical.ipynb',as_version=4)
new=[]
for cell in nb.cells:
    if cell.cell_type=='code' and cell.source.startswith('SOURCE = '):
        new.append(nbformat.v4.new_code_cell("WORK=Path(tempfile.mkdtemp(prefix='constellation-notebook-'))\nSOURCE={}\nsys.path.insert(0,str(WORK))\nOUTPUT=Path(os.environ.get('CONSTELLATION_OUTPUT',str(DATA/'outputs/notebook')));OUTPUT.mkdir(parents=True,exist_ok=True)"))
        paths=sorted(Path('constellation').glob('*.py'))+[Path(n) for n in ['run.py','audit.py','calibrate.py','geometry_oracle.py','requirements.txt']]
        for p in paths:
            source=p.read_text()
            literal="r'''"+source+"'''" if "'''" not in source else repr(source)
            new.append(nbformat.v4.new_code_cell(f"# Embedded implementation: {p}\nSOURCE[{str(p)!r}] = {literal}\npath=WORK/{str(p)!r}\npath.parent.mkdir(parents=True,exist_ok=True)\npath.write_text(SOURCE[{str(p)!r}])"))
    else:new.append(cell)
nb.cells=new
nbformat.write(nb,'Constellation_Classical.ipynb')
print('Expanded source into readable notebook cells')
