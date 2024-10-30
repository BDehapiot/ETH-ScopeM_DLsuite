#%% Imports -------------------------------------------------------------------

import time
from skimage import io
from pathlib import Path

# Functions
from bdmodel.functions import predict

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    model_path = Path.cwd() / "model_normal_768"
    # imgs_path = Path.cwd().parent / "data" / "Exp1_rf-0.1_rstack_1000-1199.tif"
    # imgs_path = Path.cwd().parent / "data" / "Exp2_rf-0.1_rstack_1000-1199.tif"
    imgs_path = Path.cwd().parent / "data" / "train_tissue" / "240611-13_4 merged_pix(13.771)_00.tif"
    
    # Open data
    imgs = io.imread(imgs_path)
    
    # Predict
    t0 = time.time()
    prds = predict(        
        imgs,
        model_path,
        img_norm="global",
        patch_overlap=0,
        )
    t1 = time.time()
    print(f"predict() : {t1 - t0:.3f}")
    
    # Display
    import napari
    import numpy as np
    prds_avg = np.mean(prds, axis=0)
    viewer = napari.Viewer()
    viewer.add_image(imgs)
    viewer.add_image(prds)
    viewer.add_image(prds_avg)