#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path

# Skimage
from skimage.measure import label

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Comments ------------------------------------------------------------------

'''
- get_edm() : 
    - works on binary images
    - inner/outer edm
    - normalize per object or not (only for inner)
    - scaling possibilities (pix/vox size required)
    - 2D or 3D or 4D...
    - return float
'''

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd().parent, "data", "train_spores") 

#%%

imgs, msks = [], []
for path in train_path.iterdir():
    if "mask" in path.stem:
        msk_path = path
        img_path = str(msk_path).replace("_mask-all", "")
        msks.append(io.imread(msk_path))
        imgs.append(io.imread(img_path))
        
def get_all(msk):
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, lab in enumerate(labels):
        tmp = msk == lab
        tmp = distance_transform_edt(tmp)
        pMax = np.percentile(tmp[tmp > 0], 99.9)
        tmp[tmp > pMax] = pMax
        tmp = (tmp / pMax)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0).astype("float32")  
    return edm
        
def get_edm(msk, direction="in", normalize="none"):
    
    global edm
    
    if not (np.issubdtype(msk.dtype, np.integer) or
            np.issubdtype(msk.dtype, np.bool_)):
        raise TypeError("Provided mask must be bool or int labels")
    
    if np.issubdtype(msk.dtype, np.bool_):
        if direction == "in":
            edm = distance_transform_edt(msk)
        elif direction == "out":
            edm = distance_transform_edt(np.invert(msk))
        if normalize == "global":
            
        
    # elif normalize == "global":
    #     pass
    # elif normalize == "object":
    #     pass
    # labels = np.unique(msk)[1:]
    
    
#%% Execute -------------------------------------------------------------------

msk = msks[0] > 0

t0 = time.time()
get_edm(msk, direction="in", normalize="none")
t1 = time.time()
print(f"get_edm() : {(t1-t0):<.5f}s")

import napari
viewer = napari.Viewer()
viewer.add_image(edm)

    
    