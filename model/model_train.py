#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd().parent, "data", "train_battery")

# Parameters
msk_suffix = ""

test = list(train_path.glob(f"*_mask{msk_suffix}*"))

#%%

def preprocess(
        train_path, 
        msk_suffix, 
        msk_type="normal", 
        normalization="global"
        ):
    
    # Nested function(s) ------------------------------------------------------
    
    def normalize(arr, pct_low=0.01, pct_high=99.99):
        return norm_pct(norm_gcn(arr)) 
    
    def _preprocess(img, msk):
                
        # Preprocess image
        if normalization == "image":
            img = normalize(img)
        
        # Preprocess mask
        if msk_type == "edt":
            msk = get_edt(msk, normalize="object", parallel=True)

        return img, msk
        
    # Execute -----------------------------------------------------------------
        
    imgs, msks = [], []
    tag = f"_mask{msk_suffix}"
    for path in train_path.iterdir():
        if tag in path.stem:
            img_name = path.name.replace(tag, "")
            imgs.append(io.imread(path.parent / img_name))
            msks.append(io.imread(path))
    imgs = np.stack(imgs)
    msks = np.stack(msks)
    
    if normalization == "global":
        imgs = normalize(imgs)
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_preprocess)(img, msk)
        for img, msk in zip(imgs, msks)
        )
    
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    return imgs, msks

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import time
    
    t0 = time.time()
    imgs, msks = preprocess(
        train_path, 
        msk_suffix, 
        msk_type="edt",
        normalization="global",
        )   
    t1 = time.time()
    print(f"preprocess() : {t1-t0:.5f}")

    import napari
    viewer = napari.Viewer()
    viewer.add_image(msks)
    
    
    
    