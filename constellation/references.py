from pathlib import Path
import cv2
import numpy as np

def extract_patterns(folder):
    result={}
    for path in sorted(Path(folder).glob('*_pattern.png')):
        rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        # Supplied diagrams encode stars as opaque white disks and edges as green.
        mask=((rgba[:,:,:3].min(axis=2)>190)&(rgba[:,:,3]>100)).astype(np.uint8)
        n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
        points=centers[1:][stats[1:,cv2.CC_STAT_AREA]>=2]
        result[path.stem.removesuffix('_pattern')]=points
    return result
