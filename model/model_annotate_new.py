#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QLabel

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd().parent, 'data', 'train_RBCs') 

# Parameters
edit = True
mask_type = "mask"
randomize = True
np.random.seed(42)
contrast_limits = (0, 65535)
brush_size = 60

#%% Function : open_image() ---------------------------------------------------

#%% Class : Painter() ---------------------------------------------------------

class Painter:
    def __init__(self, train_path, edit=True, randomize=True):
        self.train_path = train_path
        self.idx = 0
        self.init_paths()
        self.init_viewer()
        self.update()
        
    # -------------------------------------------------------------------------
        
    def update(self):
        self.open_image()
        
    def init_paths(self):
        self.img_paths = []
        for img_path in self.train_path.iterdir():
            if "mask" not in img_path.name:
                self.img_paths.append(img_path)
        if randomize:
            self.img_paths = np.random.permutation(self.img_paths).tolist()
        
    def init_viewer(self):
               
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.text_overlay.visible = True

        # Create widget
        self.widget = QWidget()
        self.layout = QVBoxLayout()

        # Create buttons
        btn_save_mask = QPushButton("Save Mask")
        btn_next_image = QPushButton("Next Image")
        btn_previous_image = QPushButton("Previous Image")

        # Create texts
        self.info = QLabel()
        self.info.setFont(QFont("Consolas", 6))
        # info.setText(update_info_text())
        
        # Add buttons and text to layout
        self.layout.addWidget(btn_save_mask)
        self.layout.addWidget(btn_next_image)
        self.layout.addWidget(btn_previous_image)
        self.layout.addSpacing(20)
        self.layout.addWidget(self.info)

        # Add layout to widget
        self.widget.setLayout(self.layout)

        # Connect buttons
        # btn_save_mask.clicked.connect(save_mask)
        btn_next_image.clicked.connect(self.next_image)
        btn_previous_image.clicked.connect(self.previous_image)

        # Add the widget to viewer
        self.viewer.window.add_dock_widget(
            self.widget, area='right', name="Painter")
        # open_image()
        # napari.run()
        
        # Shortcuts -----------------------------------------------------------
        
        @napari.Viewer.bind_key('PageUp', overwrite=True)
        def next_image_key(viewer):
            self.next_image()
            
        @napari.Viewer.bind_key('PageDown', overwrite=True)
        def previous_image_key(viewer):
            self.previous_image()
        
    # -------------------------------------------------------------------------
        
    def open_image(self):
        
        self.viewer.layers.clear()
        img_path = self.img_paths[self.idx]
        msk_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type}.tif"))
        
        if msk_path.exists() and edit:   
            img = io.imread(img_path)
            msk = io.imread(msk_path)
        elif not msk_path.exists():
            img = io.imread(img_path)
            msk = np.zeros_like(img, dtype="uint8")
            
        self.viewer.add_image(img, name="image")
        self.viewer.add_labels(msk, name="mask")
        self.viewer.layers["image"].contrast_limits = contrast_limits
        self.viewer.layers["image"].gamma = 0.66
        self.viewer.layers["mask"].brush_size = brush_size
        self.viewer.layers["mask"].selected_label = 1
        self.viewer.layers["mask"].mode = 'paint'
            
    # def save_mask(self):
    #     path = self.viewer.layers["image"].metadata["path"]
    #     path = Path(str(path).replace(".tif", f"_mask-{mask_type}.tif"))
    #     io.imsave(path, self.viewer.layers["mask"].data, check_contrast=False)  
    #     info.setText(update_info_text())
        
    def next_image(self): 
        self.idx += 1
        self.update()
        # info.setText(update_info_text())
            
    def previous_image(self):
        self.idx -= 1
        self.update()
        # info.setText(update_info_text())
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    painter = Painter(train_path, edit=edit, randomize=randomize)
