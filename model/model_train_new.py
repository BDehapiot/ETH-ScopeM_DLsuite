#%% Imports -------------------------------------------------------------------

import os
import time
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import segmentation_models as sm

# Functions
from model_functions import preprocess, augment, split

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

# Matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Comments ------------------------------------------------------------------

'''
Current:
- Manual validation split to show predictions on validation data
    - instead of creating trn_imgs and val_imgs... create indexes and pass it to model.fit
'''

'''
To Do:
- Add number of raw images inputed in training in the report 
- Manual validation split to show predictions on validation data
- Add support for multi-labels semantic segmentation (multi-class training)
- Multi-channel image support (RGB...)
- Implement starting from preexisting weights
'''

'''
To improve:
- Save model weights & associated data in dedicated folders (gitignore also)
- Dynamic vertical axis for inset plot (last n data points)
'''


#%% Class: Train() ------------------------------------------------------------

class Train:
       
    def __init__(
            self, 
            train_path,
            msk_suffix="",
            msk_type="normal",
            img_norm="global",
            patch_size=128,
            patch_overlap=32,
            nAugment=0,
            backbone="resnet18",
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            learning_rate=0.001,
            patience=20,
            ):
        
        self.train_path = train_path
        self.msk_suffix = msk_suffix
        self.msk_type = msk_type
        self.img_norm = img_norm
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.nAugment = nAugment
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        
        # Save name & path
        date = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        self.name = (f"model_{date}")
        self.save_path = Path(Path.cwd(), self.name)
        self.save_path.mkdir(exist_ok=True)
                
        # Preprocess
        self.imgs, self.msks = preprocess(
            self.train_path,
            msk_suffix=self.msk_suffix, 
            msk_type=self.msk_type, 
            img_norm=self.img_norm,
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap,
            )
        self.nImg = self.imgs.shape[0]
        
        # Augment
        os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
        if self.nAugment > 0:
            self.imgs, self.msks = augment(
                self.imgs, self.msks, self.nAugment,
                )
            
        # split
        self.trn_imgs, self.trn_msks, self.val_imgs, self.val_msks = split(
            self.imgs, self.msks, validation_split=self.validation_split)

        # Train
        self.setup()
        self.train()
        self.predict()
        self.save()
        
    # Train -------------------------------------------------------------------
        
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
            filepath=Path(self.save_path, "weights.h5"),
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True
            )
        
        # Callbacks
        self.callbacks = [
            EarlyStopping(patience=self.patience, monitor='val_loss'),
            self.checkpoint, CustomCallback(self)
            ]
    
    def train(self):
        
        self.history = self.model.fit(
            x=self.trn_imgs, y=self.trn_msks,
            validation_data=(self.val_imgs, self.val_msks),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=0,
            ) 
        
    def predict(self):
        self.val_probs = self.model.predict(self.val_imgs).squeeze()
        
    def save(self):      
               
        # History
        self.history_df = pd.DataFrame(self.history.history)
        self.history_df = self.history_df.round(5)
        self.history_df.index.name = 'Epoch'
        self.history_df.to_csv(Path(self.save_path, "history.csv"))
        
        # Report
        idx = np.argmin(self.history.history["val_loss"])
        self.report = {
            
            # Preprocess
            "train_path"       : self.train_path,
            "msk_suffix"       : self.msk_suffix,
            "msk_type"         : self.msk_type,
            "img_norm"         : self.img_norm,
            "patch_size"       : self.patch_size,
            "patch_overlap"    : self.patch_overlap,
            "nAugment"         : self.nAugment,
            
            # Train
            "backbone"         : self.backbone,
            "epochs"           : self.epochs,
            "batch_size"       : self.batch_size,
            "validation_split" : self.validation_split,
            "learning_rate"    : self.learning_rate,
            "patience"         : self.patience,
            
            # Results
            "best_epoch"       : idx,
            "best_val_loss"    : self.history.history["val_loss"][idx], 
            
            }       
        
        with open(str(Path(self.save_path, "report.txt")), "w") as f:
            for key, value in self.report.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

#%% Class: CustomCallback

class CustomCallback(Callback):
    
    def __init__(self, train):
        
        super(CustomCallback, self).__init__()
        self.train = train
        self.trn_loss, self.val_loss = [], []
        self.trn_mse, self.val_mse = [], []
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)
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
        self.ax.set_title(f"{self.train.name}")
        self.ax.set_xlabel("epochs")
        self.ax.set_ylabel("loss")
        self.ax.legend(
            loc="upper right", bbox_to_anchor=(1, -0.1), borderaxespad=0.)
                
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
        
        # Dynamic y axis
        n = 10
        if len(self.val_loss) < n: 
            y_max = np.max(self.val_loss)
        else:
            y_max = np.max(self.val_loss[-n:])
        self.axsub.set_ylim(0, y_max * 1.2)
                       
        # Info ----------------------------------------------------------------
        
        info_parameters = (
            
            f"Parameters\n"
            f"----------\n"
            f"num. of images   : {self.train.nImg}\n"
            f"num. of augment  : {self.train.nAugment}\n"
            f"Patch_size       : {self.train.patch_size}\n"
            f"backbone         : '{self.train.backbone}'\n"
            f"batch_size       : {self.train.batch_size}\n"
            f"validation_split : {self.train.validation_split}\n"
            f"learning_rate    : {self.train.learning_rate}\n"
            
            )
        
        info_monitoring = (

            f"Monitoring\n"
            f"----------\n"
            f"epoch    : {epoch + 1}/{self.train.epochs}\n"
            f"trn_loss : {logs['loss']:.4f}\n"
            f"val_loss : {logs['val_loss']:.4f} ({np.min(self.val_loss):.4f})\n"
            f"trn_mse  : {logs['loss']:.4f}\n"
            f"val_mse  : {logs['val_loss']:.4f}\n"
            f"patience : {epoch - np.argmin(self.val_loss)}/{self.train.patience}\n"
            
            )
        
        self.ax.text(
            0.00, -0.15, info_parameters,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )
       
        self.ax.text(
            0.35, -0.15, info_monitoring,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )
        
        # Draw ----------------------------------------------------------------

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Paths
    path = Path(Path.cwd().parent, "data", "train_spores")

    # Train
    train = Train(
        path,
        msk_suffix="-all",
        msk_type="edt",
        img_norm="global",
        patch_size=128,
        patch_overlap=32,
        nAugment=1000,
        backbone="resnet18",
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.001,
        patience=20,
        )
    
    val_imgs = train.val_imgs
    val_msks = train.val_msks
    val_probs = train.val_probs
    val_std = np.std(np.stack((val_msks, val_probs), axis=0), axis=0)
    
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(imgs)
    # viewer.add_image(msks)
    
    import napari
    viewer = napari.Viewer()
    viewer.add_image(val_imgs)
    viewer.add_image(val_msks)
    viewer.add_image(val_probs)
    viewer.add_image(val_std)
    
#%%

