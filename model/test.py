#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct

# Skimage
from skimage.morphology import disk, ball

#%% Comments ------------------------------------------------------------------

'''
General cases:
    
    1) raw image + binary mask
    2) raw image + labeled mask (semantic)
    3) raw image + labeled mask (instance)
    
Function to test:
    - preprocess()
        - input imgs ("uint8", "uint16", "float") 
        - input msks ("uint8", "uint16", "float")
    - augment()
    - predict()
    
'''

#%% Function(s) ---------------------------------------------------------------

def generate_random_data(
        nZ, nY, nX, 
        nObj, min_radius, max_radius,
        noise_sd, dtype
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
    img += np.random.normal(0, noise_sd, img.shape)
    img = norm_pct(img) * dtype
    
    return img, msk

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    maxInt = 255
    nZ, nY, nX, nObj = 1, 512, 512, 8
    min_radius, max_radius = 8, 16
    
    msk = generate_random_mask(
        nZ, nY, nX, nObj, min_radius, max_radius)
    img = (msk > 0).astype(float)
    img += np.random.normal(0, 0.5, img.shape)
    img = norm_pct(img) * maxInt
    
    # Display
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    # viewer.add_labels(msk)