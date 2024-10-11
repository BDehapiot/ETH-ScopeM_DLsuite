#%% Imports -------------------------------------------------------------------

import sys
import shutil
import pytest
import numpy as np
from skimage import io
from pathlib import Path

# model_function
from bdmodel.functions import preprocess

# bdtools
from bdtools.norm import norm_pct

# Skimage
from skimage.morphology import disk, ball

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'patch'
sys.path.insert(0, str(ROOT_PATH))

#%% Function(s) ---------------------------------------------------------------

def random_data(
        nData,
        nZ, nY, nX, 
        nObj, min_radius, max_radius,
        img_noise, img_dtype, msk_dtype,
        ):
       
    # Nested function(s) ------------------------------------------------------
    
    def _random_data(
            nZ, nY, nX, 
            nObj, min_radius, max_radius,
            img_noise, img_dtype, msk_dtype
            ):
            
        # Define random variables
        zIdx = np.random.randint(0, nZ, nObj)
        yIdx = np.random.randint(0, nY, nObj)
        xIdx = np.random.randint(0, nX, nObj)
        if min_radius >= max_radius:
            min_radius -= 1
        radius = np.random.randint(min_radius, max_radius, nObj)
        labels = np.random.choice(
            np.arange(1, nObj * 2), size=nObj, replace=False)
        
        # Create mask
        msk = []
        for i in range(nObj):
            tmp = np.zeros((nZ, nY, nX), dtype="int32").squeeze()
            
            if nZ > 1:
                
                obj = ball(radius[i])
                z0 = zIdx[i] - obj.shape[0] // 2
                y0 = yIdx[i] - obj.shape[1] // 2
                x0 = xIdx[i] - obj.shape[2] // 2
                z1 = z0 + obj.shape[0]
                y1 = y0 + obj.shape[1]
                x1 = x0 + obj.shape[2]
                if z0 < 0:
                    obj = obj[-z0:, :, :]; z0 = 0
                if z1 > nZ:
                    obj = obj[:nZ - z0, :, :]; z1 = nZ
                if y0 < 0:  
                    obj = obj[:, -y0:, :]; y0 = 0
                if y1 > nY: 
                    obj = obj[:, :nY - y0, :]; y1 = nY
                if x0 < 0:  
                    obj = obj[:, :, -x0:]; x0 = 0
                if x1 > nX: 
                    obj = obj[:, :, :nX - x0]; x1 = nX
                tmp[z0:z1, y0:y1, x0:x1] = obj
            
            else:
                
                obj = disk(radius[i])
                y0 = yIdx[i] - obj.shape[0] // 2
                x0 = xIdx[i] - obj.shape[1] // 2
                y1 = y0 + obj.shape[0]
                x1 = x0 + obj.shape[1]
                if y0 < 0:  
                    obj = obj[-y0:, :]; y0 = 0
                if y1 > nY: 
                    obj = obj[:nY - y0, :]; y1 = nY
                if x0 < 0:  
                    obj = obj[:, -x0:]; x0 = 0
                if x1 > nX: 
                    obj = obj[:, :nX - x0]; x1 = nX
                tmp[y0:y1, x0:x1] = obj
            
            tmp *= labels[i]
            msk.append(tmp)
            
        if msk:
            msk = msk[0] if nObj == 1 else np.max(np.stack(msk), axis=0)
        else:
            msk = np.zeros((nZ, nY, nX), dtype="int32").squeeze()
        
        # Create associated image
        img = (msk > 0).astype(float)
        img += np.random.normal(0, img_noise, img.shape)
        img = norm_pct(img)
        if img_dtype == "uint8" : 
            img = (img * 255).astype("uint8")
        if img_dtype == "uint16" :
            img = (img * 65535).astype("uint16")
        
        return img, msk
    
    # Execute -----------------------------------------------------------------
    
    msks, imgs = [], []
    for n in range(nData):
        img, msk = _random_data(
            nZ, nY, nX, 
            nObj, min_radius, max_radius,
            img_noise, img_dtype, msk_dtype
            )
        imgs.append(img)
        msks.append(msk)
        
    return imgs, msks

#%% Test cases ----------------------------------------------------------------

params_random_data = []
params_preprocess = []
for i in range(10):
    
    # random_data() parameters
    nData = np.random.randint(5, 20)
    nZ = np.random.choice([1, np.random.randint(2, 32)])
    nY = np.random.randint(64, 512)
    nX = np.random.randint(64, 512)
    nObj = np.random.randint(0, 16)
    min_radius = np.random.randint(8, 16)
    max_radius = round(min_radius * np.random.uniform(1.1, 3))
    img_noise = round(np.random.uniform(0.5, 1.0), 3)
    img_dtype = str(np.random.choice(["uint8", "uint16", "float32"]))
    msk_dtype = str(np.random.choice(["uint8", "uint16", "float32", "bool"]))

    # # preprocess() parameters
    # msk_type = str(np.random.choice(["normal", "edt", "bounds"]))
    # img_norm = str(np.random.choice(["none", "global", "image"]))
    # patch_size = np.random.choice([0, np.random.randint(16, 64)])
    # if patch_size == 0:
    #     patch_overlap = 0 
    # else:
    #     patch_overlap = np.random.choice(
    #         [0, np.random.randint(8, patch_size - 1)])
            
    params_random_data.append((
        nData,
        nZ, nY, nX,
        nObj, min_radius, max_radius,
        img_noise, img_dtype, msk_dtype,
        )) 
    
    # params_preprocess.append((
    #     img_norm, msk_type,
    #     patch_size, patch_overlap,
    #     )) 
    
#%% Tests ---------------------------------------------------------------------
   
# Generate random data 
for i, param in enumerate(params_random_data):
    imgs, msks = random_data(        
        nData,
        nZ, nY, nX, 
        nObj, min_radius, max_radius,
        img_noise, img_dtype, msk_dtype,
        )

# @pytest.mark.parametrize(
#     "nData, "
#     "nZ, nY, nX, "
#     "nObj, min_radius, max_radius, "
#     "img_noise, img_dtype, msk_dtype, ",
#     params_random_data
#     )

# def test_random_data(
#         nData,
#         nZ, nY, nX,
#         nObj, min_radius, max_radius,
#         img_noise, img_dtype, msk_dtype,
#         ):
    
#     # Genrate random data
#     imgs, msks = random_data(        
#         nData,
#         nZ, nY, nX, 
#         nObj, min_radius, max_radius,
#         img_noise, img_dtype, msk_dtype,
#         )
    
#     # Save
#     train_path = Path.cwd() / "train"
#     if train_path.exists():
#         shutil.rmtree(train_path)
#     train_path.mkdir(exist_ok=True)
#     for i, (img, msk) in enumerate(zip(imgs, msks)):
#         io.imsave(
#             train_path / f"img_{i:02d}.tif",
#             img, check_contrast=False,
#             )
#         io.imsave(
#             train_path / f"msk_{i:02d}.tif",
#             msk, check_contrast=False,
#             )
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])