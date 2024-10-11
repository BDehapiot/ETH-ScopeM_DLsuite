#%% Imports -------------------------------------------------------------------

import time
import pickle
from skimage import io
from pathlib import Path
import segmentation_models as sm

# Functions
from bdmodel.functions import preprocess

# bdtools
from bdtools.patch import merge_patches

#%% Comments ------------------------------------------------------------------

'''
- Predict larger than VRAM 
'''

#%% Function: predict() -------------------------------------------------------

def predict(
        imgs, 
        model_path, 
        img_norm="global",
        patch_overlap=0,
        ):

    valid_norms = ["none", "global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
        
    # Nested function(s) ------------------------------------------------------
        
    # Execute -----------------------------------------------------------------
    
    # Load report
    with open(str(model_path / "report.pkl"), "rb") as f:
        report = pickle.load(f)
    
    # Load model
    model = sm.Unet(
        report["backbone"], 
        input_shape=(None, None, 1), 
        classes=1, 
        activation="sigmoid", 
        encoder_weights=None,
        )
    
    # Load weights
    model.load_weights(model_path / "weights.h5") 
       
    # Preprocess
    patches = preprocess(
        imgs, msks=None, 
        img_norm=img_norm,
        patch_size=report["patch_size"], 
        patch_overlap=patch_overlap,
        )

    # Predict
    prds = model.predict(patches).squeeze()
    prds = merge_patches(prds, imgs.shape, patch_overlap)
    
    return prds

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    model_path = Path.cwd() / "model_normal"
    # imgs_path = Path.cwd().parent / "data" / "Exp1_rf-0.1_rstack_1000-1199.tif"
    imgs_path = Path.cwd().parent / "data" / "Exp2_rf-0.1_rstack_1000-1199.tif"
    
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
    
    