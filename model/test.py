import napari
import numpy as np
from skimage import io
from pathlib import Path

from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries 

msk_path = Path("R20_003_020_C1_mask-mask.tif") 
msk = io.imread(msk_path)

#%%

#%%

msk_obj = label(msk > 0 ^ find_boundaries(msk), connectivity=1)

nObjects = np.maximum(0, len(np.unique(msk_obj)) - 1)
nLabels = np.maximum(0, len(np.unique(msk)) - 1)
minLabel = np.min(msk)
maxLabel = np.max(msk)

# Get missing labels (between min & max label)
missLabels = []
for lbl in range(maxLabel):
    if np.all(msk != lbl):
        missLabels.append(lbl)

# Get duplicated labels (multi objects label)
lbls = []
for props in regionprops(msk_obj, intensity_image=msk):
    lbls.append(int(props.intensity_max))
uniq, count = np.unique(lbls, return_counts=True)
dupLabels = []
for l, lbl in enumerate(uniq):
    if count[l] > 1:
        dupLabels.append(f"{lbl}({count[l]})")
dupLabels = " ".join(dupLabels)

print(f"missLabels{missLabels}")
print(uniq)
print(count)
print(dupLabels)

#%% 

from skimage.segmentation import expand_labels

msk_obj = label(msk > 0 ^ find_boundaries(msk), connectivity=1)
msk_obj = expand_labels(msk_obj)
msk_obj[msk == 0] = 0

#%%

viewer = napari.Viewer()
viewer.add_labels(msk)
viewer.add_labels(msk_obj)