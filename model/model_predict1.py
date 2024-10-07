#%% Imports -------------------------------------------------------------------

import pickle
from skimage import io
from pathlib import Path
import segmentation_models as sm

# bdtools
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches, merge_patches

#%% Comments ------------------------------------------------------------------

#%% Function: predict() -------------------------------------------------------

def predict(imgs, model_path, img_norm="global"):
    
    valid_norms = ["none", "global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
        
    # Nested function(s) ------------------------------------------------------
    
    def normalize(arr, pct_low=0.01, pct_high=99.99):
        return norm_pct(norm_gcn(arr)) 
    
    # def _preprocess(img):
    #     if img_norm == "image":
    #         img = normalize(img)     
    #     if patch_size > 0:
    #         img = extract_patches(img, patch_size, patch_overlap)
    #     return img
    
    # Execute -----------------------------------------------------------------
    
    # Parse report
    with open(str(model_path / "report.pkl"), "rb") as f:
        report = pickle.load(f)
    backbone = report["backbone"]
    patch_size = report["patch_size"]
    patch_overlap = report["patch_overlap"]
    
    # Preprocessing
    if img_norm == "global":
        imgs = normalize(imgs)
    # imgs = extract_patches(imgs, patch_size, patch_overlap)
    
        
    # Model
    model = sm.Unet(
        backbone, 
        input_shape=(None, None, 1), 
        classes=1, 
        activation="sigmoid", 
        encoder_weights=None,
        )

    pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    model_path = Path.cwd() / "model_edt"
    imgs_path = Path.cwd().parent / "data" / "Exp1_rf-0.1_stack.tif"
    
    # 
    imgs = io.imread(imgs_path)
    
    #
    predict(imgs, model_path)