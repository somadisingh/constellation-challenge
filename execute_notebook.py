import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
os.environ['CONSTELLATION_DATA']=str(Path('.').resolve())
os.environ['CONSTELLATION_MODE']='smoke'
os.environ['CONSTELLATION_SKIP_INSTALL']='1'
os.environ['CONSTELLATION_OUTPUT']=str(Path('outputs/notebook_smoke').resolve())
os.environ['IPYTHONDIR']=str(Path('outputs/ipython').resolve())
nb=nbformat.read('Constellation_Classical.ipynb',as_version=4)
NotebookClient(nb,timeout=600,kernel_name='python3').execute()
nbformat.write(nb,'Constellation_Classical.executed.ipynb')
print('Notebook completed top-to-bottom in smoke mode')
