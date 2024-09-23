#%% Imports -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path

# bdtools
from bdtools.mask import get_edt

#%%

train_path = Path(Path.cwd().parent, "data", "train_battery")
msk = io.imread(train_path / "CBS Image_CBS-Custom_q1_x01_y01_s0049_0006_0057_mask.tif")
edt = get_edt(msk, normalize="object")

viewer = napari.Viewer()
viewer.add_image(edt)
