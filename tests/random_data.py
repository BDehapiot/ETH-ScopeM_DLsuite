#%% Imports -------------------------------------------------------------------

import numpy as np

# bdtools
from bdtools.norm import norm_pct

# Skimage
from skimage.morphology import disk, ball

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

params = [
    
    # 
    {
     "nData" : 16,
     "nZ" : 1, "nY" : 512, "nX" : 512,
     "nObj" : 16, "min_radius" : 16, "max_radius" : 32,
     "img_noise" : 0.5, "img_dtype" : "uint8", "msk_dtype" : "bool",
    },
    
    ]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for i in range(10):
        msks, imgs = 