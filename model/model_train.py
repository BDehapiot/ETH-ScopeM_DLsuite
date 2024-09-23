#%% Imports -------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import albumentations as A
from datetime import datetime
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches

# Skimage
from skimage.segmentation import find_boundaries 

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#%% Comments ------------------------------------------------------------------

'''
- Save model weights & associated data in dedicated folders (gitignore also)
- Set live update (log & plot see chatGPT)
- Add support for multi-labels semantic segmentation (multi-class training)
- Make a wrapper function for the full process 
'''

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd().parent, "data", "train_spores")

# Parameters
msk_suffix = "-all"
msk_type = "edt"
patch_size = 128
patch_overlap = 32
iterations = 100

#%% Function: preprocess() ----------------------------------------------------
   
def preprocess(
        train_path, 
        msk_suffix="", 
        msk_type="normal", 
        img_norm="global",
        patch_size=0, 
        patch_overlap=0,
        ):
    
    valid_types = ["normal", "edt", "bounds"]
    if msk_type not in valid_types:
        raise ValueError(
            f"Invalid value for msk_type: '{msk_type}'."
            f" Expected one of {valid_types}."
            )

    valid_norms = ["none", "global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
    
    # Nested function(s) ------------------------------------------------------
    
    def open_data(train_path, msk_suffix):
        imgs, msks = [], []
        tag = f"_mask{msk_suffix}"
        for path in train_path.iterdir():
            if tag in path.stem:
                img_name = path.name.replace(tag, "")
                imgs.append(io.imread(path.parent / img_name))
                msks.append(io.imread(path))
        imgs = np.stack(imgs)
        msks = np.stack(msks)
        return imgs, msks
    
    def normalize(arr, pct_low=0.01, pct_high=99.99):
        return norm_pct(norm_gcn(arr)) 
    
    def _preprocess(img, msk):
                
        if img_norm == "image":
            img = normalize(img)
        
        if msk_type == "normal":
            msk = msk > 0
        elif msk_type == "edt":
            msk = get_edt(msk, normalize="object", parallel=False)
        elif msk_type == "bounds":
            msk = find_boundaries(msk)           

        if patch_size > 0:
            img = extract_patches(img, patch_size, patch_overlap)
            msk = extract_patches(msk, patch_size, patch_overlap)

        return img, msk
        
    # Execute -----------------------------------------------------------------

    imgs, msks = open_data(train_path, msk_suffix)

    if normalize == "global":
        imgs = normalize(imgs)
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_preprocess)(img, msk)
        for img, msk in zip(imgs, msks)
        )    
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    if patch_size > 0:
        imgs = np.stack([arr for sublist in imgs for arr in sublist])
        msks = np.stack([arr for sublist in msks for arr in sublist])
    
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

#%% Function: train() ---------------------------------------------------------

def train(
        imgs, msks,
        backbone="resnet18", # resnet 18, 34, 50, 101 or 152 
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.001,
        patience=20,
        ):
    
    path = (
        f"model-weights_"
        f"mask{msk_suffix}-{msk_type}_{patch_size}_"
        f"{datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')}.h5"
        )
    
    # Setup model -------------------------------------------------------------
   
    model = sm.Unet(
        backbone, 
        input_shape=(None, None, 1), 
        classes=1, 
        activation="sigmoid", 
        encoder_weights=None,
        )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy", 
        metrics=["mse"],
        )
    
    # Checkpoints & callbacks -------------------------------------------------
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True
        )
    callbacks = [
        EarlyStopping(patience=patience, monitor='val_loss'),
        model_checkpoint_callback
        ]
    
    # Train model -------------------------------------------------------------
    
    history = model.fit(
        x=imgs, y=msks,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        )
    
    # Plot & save -------------------------------------------------------------
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    train_data = pd.DataFrame(history.history)
    train_data = train_data.round(5)
    train_data.index.name = 'Epoch'
    train_data.to_csv(path.replace(".h5", ".csv"))

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import time
        
    t0 = time.time()
    imgs, msks = preprocess(
        train_path, 
        msk_suffix=msk_suffix, 
        msk_type="edt", 
        img_norm="global",
        patch_size=256, 
        patch_overlap=32,
        )
    t1 = time.time()
    print(f"preprocess() : {t1-t0:.5f}")
    
    t0 = time.time()
    imgs, msks = augment(imgs, msks, iterations)
    t1 = time.time()
    print(f"augment() : {t1-t0:.5f}")
    
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(imgs)
    # viewer.add_image(msks)
    
    t0 = time.time()
    train(        
        imgs, msks,
        backbone="resnet18", # ResNet 18, 34, 50, 101 or 152 
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.001,
        patience=20,
        )
    t1 = time.time()
    print(f"train() : {t1-t0:.5f}")
  
    