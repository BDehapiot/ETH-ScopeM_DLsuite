#%% Imports -------------------------------------------------------------------

import os
import pickle
import warnings
import numpy as np
from skimage import io
from pathlib import Path
from matplotlib import cm
os.environ['NO_ALBUMENTATIONS_UPDATE'] = "1" # Don't know if it works
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches, merge_patches

# Skimage
from skimage.segmentation import find_boundaries 

#%% Functions: ----------------------------------------------------------------

def open_data(train_path, msk_suffix):
    imgs, msks = [], []
    tag = f"_mask{msk_suffix}"
    for path in train_path.iterdir():
        if tag in path.stem:
            img_name = path.name.replace(tag, "")
            imgs.append(io.imread(path.parent / img_name))
            msks.append(io.imread(path))
    return imgs, msks

def split_idx(n, validation_split=0.2):
    val_n = int(n * validation_split)
    trn_n = n - val_n
    idx = np.arange(n)
    np.random.shuffle(idx)
    trn_idx = idx[:trn_n]
    val_idx = idx[trn_n:]
    return trn_idx, val_idx

def save_val_prds(imgs, msks, prds, save_path):

    plt.ioff() # turn off inline plot
    
    for i in range(imgs.shape[0]):

        # Initialize
        fig, (ax0, ax1, ax2) = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5))
        cmap0, cmap1, cmap2 = cm.gray, cm.plasma, cm.plasma
        shrink = 0.75

        # Plot img
        ax0.imshow(imgs[i], cmap=cmap0)
        ax0.set_title("image")
        ax0.set_xlabel("pixels")
        ax0.set_ylabel("pixels")
        fig.colorbar(
            cm.ScalarMappable(cmap=cmap0), ax=ax0, shrink=shrink)

        # Plot msk
        ax1.imshow(msks[i], cmap=cmap1)
        ax1.set_title("mask")
        ax1.set_xlabel("pixels")
        fig.colorbar(
            cm.ScalarMappable(cmap=cmap1), ax=ax1, shrink=shrink)
        
        # Plot prd
        ax2.imshow(prds[i], cmap=cmap2)
        ax2.set_title("prediction")
        ax2.set_xlabel("pixels")
        fig.colorbar(
            cm.ScalarMappable(cmap=cmap2), ax=ax2, shrink=shrink)
        
        plt.tight_layout()
        
        # Save
        Path(save_path, "val_prds").mkdir(exist_ok=True)
        plt.savefig(save_path / "val_prds" / f"expl_{i:02d}.png")
        plt.close(fig)

#%% Function: preprocess() ----------------------------------------------------
   
def preprocess(
        imgs, msks=None,
        img_norm="global",
        msk_type="normal", 
        patch_size=256, 
        patch_overlap=0,
        ):
    
    """ 
    Preprocess images and masks for training or prediction procedures.
    
    If msks=None, only images will be preprocessed.
    Images and masks will be splitted into patches.
    
    Parameters
    ----------
    imgs : 2D ndarray or list of 2D ndarrays (int or float)
        Inputs image(s).
        
    msks : 2D ndarray or list of 2D ndarrays (bool or int), optional, default=None 
        Inputs mask(s).
        If None, only images will be preprocessed.
        
    img_norm : str, default="global"
        - "global" : 0 to 1 normalization considering the full stack.
        - "image"  : 0 to 1 normalization per image.
        
    msk_type : str, default="normal"
        - "normal" : No changes.
        - "edt"    : Euclidean distance transform of binary/labeled objects.
        - "bounds" : Boundaries of binary/labeled objects.

    patch_size : int, default=256
        Size of extracted patches.
        Should be int > 0 and multiple of 2.
    
    patch_overlap : int, default=0
        Overlap between patches.
        Should be int, from 0 to patch_size - 1.
        
    Returns
    -------  
    imgs : 3D ndarray (float32)
        Preprocessed images.
        
    msks : 3D ndarray (float32), optional
        Preprocessed masks.
        
    """
    
    valid_types = ["normal", "edt", "bounds"]
    if msk_type not in valid_types:
        raise ValueError(
            f"Invalid value for msk_type: '{msk_type}'."
            f" Expected one of {valid_types}."
            )

    valid_norms = ["global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
        
    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError(
            f"Invalid value for patch_size: '{patch_size}'."
            f" Should be int > 0 and multiple of 2."
            )

    if patch_overlap < 0 or patch_overlap >= patch_size:
        raise ValueError(
            f"Invalid value for patch_overlap: '{patch_overlap}'."
            f" Should be int, from 0 to patch_size - 1."
            )

    # Nested function(s) ------------------------------------------------------

    def normalize(arr, pct_low=0.01, pct_high=99.99):
        return norm_pct(norm_gcn(arr), pct_low=pct_low, pct_high=pct_high)      
            
    def _preprocess(img, msk=None):

        if msks is None:
            
            img = np.array(img)
            
            img = extract_patches(img, patch_size, patch_overlap)
                 
            return img
            
        else:
            
            img = np.array(img)
            msk = np.array(msk)
            
            if msk_type == "normal":
                msk = msk > 0
            elif msk_type == "edt":
                msk = get_edt(msk, normalize="object", parallel=False)
            elif msk_type == "bounds":
                msk = find_boundaries(msk)           
            
            img = extract_patches(img, patch_size, patch_overlap)
            msk = extract_patches(msk, patch_size, patch_overlap)
                
            return img, msk
    
    # Execute -----------------------------------------------------------------        
       
    # Normalize images
    if img_norm == "global":
        imgs = normalize(imgs)
    if img_norm == "image":
        imgs = [normalize(img) for img in imgs]
   
    # Preprocess
    if msks is None:
        
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        
        if len(imgs) > 1:
               
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img)
                for img in imgs
                )
            imgs = [data[0] for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
                
        else:
            
            imgs = _preprocess(imgs)
            imgs = np.stack(imgs)
        
        imgs = imgs.astype("float32")
        
        return imgs
    
    else:
        
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        
        if len(imgs) > 1:
            
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img, msk)
                for img, msk in zip(imgs, msks)
                )
            imgs = [data[0] for data in outputs]
            msks = [data[1] for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
            msks = np.stack([arr for sublist in msks for arr in sublist])
            
        else:
            
            imgs, msks = _preprocess(imgs, msks)
            imgs = np.stack(imgs)
            msks = np.stack(msks)
            
        imgs = imgs.astype("float32")
        msks = msks.astype("float32")
        
        return imgs, msks
    
#%% Function: augment() -------------------------------------------------------

def augment(imgs, msks, iterations):
        
    if iterations <= imgs.shape[0]:
        warnings.warn(f"iterations ({iterations}) is less than n of images")
        
    # Nested function(s) ------------------------------------------------------
    
    def _augment(imgs, msks, operations):      
        idx = np.random.randint(0, len(imgs) - 1)
        outputs = operations(image=imgs[idx,...], mask=msks[idx,...])
        return outputs["image"], outputs["mask"]
    
    # Execute -----------------------------------------------------------------
    
    operations = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(p=0.5),
        ])
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_augment)(imgs, msks, operations)
        for i in range(iterations)
        )
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    return imgs, msks

#%% Function: predict() -------------------------------------------------------

'''
- Predict data larger than VRAM
'''

def predict(
        imgs, 
        model_path, 
        img_norm="global",
        patch_overlap=0,
        ):
    
    global prds

    valid_norms = ["global", "image"]
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