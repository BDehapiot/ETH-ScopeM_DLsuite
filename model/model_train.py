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
- Save model weights & associated data in dedicated folders (gitignore also)
- Set live update (log & plot see chatGPT)
- Add support for multi-labels semantic segmentation (multi-class training)
- Make a wrapper function for the full process 
- Implement starting from preexisting weights
'''

#%% Function: train() ---------------------------------------------------------

def train(
        imgs, msks,
        epochs=50,
        backbone="resnet18", # resnet 18, 34, 50, 101 or 152
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.001,
        patience=20,
        ):
    
    name = (
        f"model-weights_"
        f"mask{msk_suffix}-{msk_type}_{patch_size}_"
        f"{datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')}.h5"
        )
    
    shape = imgs.shape
    
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
        filepath=name,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True
        )
    callbacks = [
        EarlyStopping(patience=patience, monitor='val_loss'),
        model_checkpoint_callback, CustomCallback(
            epochs, 
            backbone, 
            batch_size, 
            validation_split, 
            learning_rate, 
            patience,
            name,
            
            ),
        ]
    
    # Train model -------------------------------------------------------------
    
    history = model.fit(
        x=imgs, y=msks,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
        )
    
    # Plot & save -------------------------------------------------------------
    
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'y', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # train_data = pd.DataFrame(history.history)
    # train_data = train_data.round(5)
    # train_data.index.name = 'Epoch'
    # train_data.to_csv(name.replace(".h5", ".csv"))

#%% Class: CustomCallback() ---------------------------------------------------

class CustomCallback(Callback):
    def __init__(
            self, 
            epochs, 
            backbone, 
            batch_size, 
            validation_split, 
            learning_rate, 
            patience,
            name,
            ):
        
        super(CustomCallback, self).__init__()
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        self.name = name
        self.trn_loss = []
        self.val_loss = []
        self.trn_mse = []
        self.val_mse = []
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.4)
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
  
    