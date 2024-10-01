#%% Imports -------------------------------------------------------------------

import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import segmentation_models as sm

# Functions
from model_functions import preprocess, augment

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

# Matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Comments ------------------------------------------------------------------

'''
- Make augmentation optional
- Save model weights & associated data in dedicated folders (gitignore also)
- Add support for multi-labels semantic segmentation (multi-class training)
- Implement starting from preexisting weights
'''

#%% Class: Train() ------------------------------------------------------------

class Train:
    
    def __init__(
            self, 
            path,
            msk_suffix="",
            msk_type="normal",
            img_norm="global",
            patch_size=128,
            patch_overlap=32,
            iterations=100,
            backbone="resnet18",
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            learning_rate=0.001,
            patience=20,
            ):
        
        self.path = path
        self.msk_suffix = msk_suffix
        self.msk_type = msk_type
        self.img_norm = img_norm
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.iterations = iterations
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
                
        # Preprocess
        self.imgs, self.msks = preprocess(
            self.path,
            msk_suffix=self.msk_suffix, 
            msk_type=self.msk_type, 
            img_norm=self.img_norm,
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap,
            )
        
        # Augment
        self.imgs, self.msks = augment(
            self.imgs, self.msks, self.iterations,
            )

        # Train
        self.setup()
        self.train()
        
    # -------------------------------------------------------------------------
        
    def setup(self):
        
        # Model
        
        self.model = sm.Unet(
            self.backbone, 
            input_shape=(None, None, 1), 
            classes=1, 
            activation="sigmoid", 
            encoder_weights=None,
            )
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy", 
            metrics=["mse"],
            )
        
        # Checkpoint
        self.checkpoint = ModelCheckpoint(
            filepath=self.model_name,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True
            )
        
        # Callbacks
        self.callbacks = [
            EarlyStopping(patience=self.patience, monitor='val_loss'),
            self.checkpoint, CustomCallback(
                self.epochs, 
                self.backbone, 
                self.batch_size, 
                self.validation_split, 
                self.learning_rate, 
                self.patience,
                self.model_name,
                ),
            ]
    
    def train(self):
        
        self.history = self.model.fit(
            x=self.imgs, y=self.msks,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=0,
            ) 

#%% Class: CustomCallback

class CustomCallback(Callback):
    
    def __init__(
            self, 
            epochs, 
            backbone, 
            batch_size, 
            validation_split, 
            learning_rate, 
            patience,
            model_name,
            ):
        
        super(CustomCallback, self).__init__()
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        self.model_name = model_name
        self.trn_loss = []
        self.val_loss = []
        self.trn_mse = []
        self.val_mse = []
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
        self.axsub = None
        
        plt.rcParams["font.family"] = "Consolas"
        plt.rcParams["font.size"] = 12
        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        
        # Get loss and mse values
        self.trn_loss.append(logs["loss"])
        self.val_loss.append(logs.get("val_loss"))
        self.trn_mse.append(logs["mse"])
        self.val_mse.append(logs.get("val_mse"))

        # Main plot -----------------------------------------------------------
        
        self.ax.clear()
        self.ax.plot(
            range(1, epoch + 2), self.trn_loss, "y", label="training loss")
        self.ax.plot(
            range(1, epoch + 2), self.val_loss, "r", label="validation loss")
        self.ax.set_title("loss")
        self.ax.set_xlabel("epochs")
        self.ax.set_ylabel("loss")
        self.ax.legend(loc="upper left")
                
        # Subplot -------------------------------------------------------------
        
        if self.axsub is not None:
            self.axsub.clear()
        else:
            self.axsub = inset_axes(
                self.ax, width="50%", height="50%", loc="upper right")
        self.axsub.plot(
            range(1, epoch + 2), self.trn_loss, "y", label="training loss")
        self.axsub.plot(
            range(1, epoch + 2), self.val_loss, "r", label="validation loss")
        self.axsub.set_xlabel("epochs")
        self.axsub.set_ylabel("loss")
        self.axsub.set_ylim(0, 0.2)
                       
        # Info ----------------------------------------------------------------

        self.vloss_best = np.min(self.val_loss)
        self.epoch_best = np.argmin(self.val_loss)

        info_name = (
            
            f"{self.name}"
            
            )
        
        info_parameters = (
            
            f"Parameters\n"
            f"----------\n"
            f"input shape      : {self.patience}\n"
            f"epochs           : {self.epochs}\n"
            f"backbone         : '{self.backbone}'\n"
            f"batch_size       : {self.batch_size}\n"
            f"validation_split : {self.validation_split}\n"
            f"learning_rate    : {self.learning_rate}\n"
            
            )
        
        info_monitoring = (

            f"Monitoring\n"
            f"----------\n"
            f"epoch    : {epoch + 1}/{self.epochs}\n"
            f"trn_loss : {logs['loss']:.4f}\n"
            f"val_loss : {logs['val_loss']:.4f} ({self.vloss_best:.4f})\n"
            f"trn_mse  : {logs['loss']:.4f}\n"
            f"val_mse  : {logs['val_loss']:.4f}\n"
            f"patience : {epoch - self.epoch_best}/{self.patience}\n"
            
            )
        
        self.ax.text(
            0.00, -0.2, info_name,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )

        self.ax.text(
            0.00, -0.3, info_parameters,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )
       
        self.ax.text(
            0.35, -0.3, info_monitoring,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )
        
        # Draw ----------------------------------------------------------------

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import time
    
    # Paths
    train_path = Path(Path.cwd().parent, "data", "train_spores")

    # Parameters
    msk_suffix = "-all"
    msk_type = "edt"
    patch_size = 128
    patch_overlap = 32
    iterations = 100
        
    # preprocess()
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
    
    # augment()
    t0 = time.time()
    imgs, msks = augment(imgs, msks, iterations)
    t1 = time.time()
    print(f"augment() : {t1-t0:.5f}")
    
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(imgs)
    # viewer.add_image(msks)
    
    # Train()
    train = Train(imgs, msks)