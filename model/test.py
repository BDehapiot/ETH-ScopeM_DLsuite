import napari
import numpy as np
from skimage import io
from pathlib import Path

msk_path = Path("CBS Image_CBS-Custom_q1_x01_y01_s0575_0532_0079_mask-mask.tif") 
msk = io.imread(msk_path)

#%%

from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_dilation
from skimage.measure import label, regionprops

msk_bin0 = msk > 0
msk_bin1 = binary_fill_holes(msk_bin0) 
msk_bin2 = msk_bin1 ^ msk_bin0
msk_bin2 = binary_dilation(msk_bin2)
for props in regionprops(label(msk_bin2), intensity_image=msk):
    idx = tuple((props.coords[:, 0], props.coords[:, 1]))
    label = int(props.max_intensity)
    msk[idx] = label

#%%

viewer = napari.Viewer()
viewer.add_labels(msk)
# viewer.add_labels(msk_bin2)
