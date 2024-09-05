#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# Qt
from qtpy.QtGui import QFont
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QLabel

#%% Comments ------------------------------------------------------------------

'''
- To reduce loading time preload images in a list? Could be an optionnal
- Get statistics, updated live or when saving mask?

'''

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd().parent, 'data', 'train_spores') 

# Parameters
edit = True
mask_type = "mask"
randomize = True
np.random.seed(42)
contrast_limits = (0, 1)
brush_size = 10

#%% Class : Painter() ---------------------------------------------------------

class Painter:
    def __init__(self, train_path, edit=True, randomize=True):
        self.train_path = train_path
        self.idx = 0
        self.init_paths()
        self.init_images()
        self.init_viewer()
        self.update()
        
        # Timers
        self.next_brush_size_timer = QTimer()
        self.next_brush_size_timer.timeout.connect(self.next_brush_size)
        self.prev_brush_size_timer = QTimer()
        self.prev_brush_size_timer.timeout.connect(self.prev_brush_size)

    def update(self):
        self.get_info_text()
        self.open_image()
        
#%% Initialize ----------------------------------------------------------------
        
    def init_paths(self):
        self.img_paths = []
        for img_path in self.train_path.iterdir():
            if "mask" not in img_path.name:
                self.img_paths.append(img_path)
        if randomize:
            self.img_paths = np.random.permutation(self.img_paths).tolist()
            
    def init_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            self.imgs.append(io.imread(img_path))
        
    def init_viewer(self):
               
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.text = ""

        # Create widget
        self.widget = QWidget()
        self.layout = QVBoxLayout()

        # Create buttons
        btn_save_mask = QPushButton("Save Mask")
        btn_next_image = QPushButton("Next Image")
        btn_prev_image = QPushButton("Previous Image")

        # Create texts
        self.info = QLabel()
        self.info.setFont(QFont("Consolas", 6))
        
        # Add buttons and text to layout
        self.layout.addWidget(btn_save_mask)
        self.layout.addWidget(btn_next_image)
        self.layout.addWidget(btn_prev_image)
        self.layout.addSpacing(20)
        self.layout.addWidget(self.info)

        # Add layout to widget
        self.widget.setLayout(self.layout)

        # Connect buttons
        btn_save_mask.clicked.connect(self.save_mask)
        btn_next_image.clicked.connect(self.next_image)
        btn_prev_image.clicked.connect(self.prev_image)

        # Add the widget to viewer
        self.viewer.window.add_dock_widget(
            self.widget, area='right', name="Painter")
        
#%% Shortcuts -----------------------------------------------------------------
        
        @self.viewer.bind_key("End", overwrite=True)
        def save_mask_key(viewer):
            self.save_mask() 
            
        @self.viewer.bind_key('Down', overwrite=True)
        def prev_label_key(viewer):
            self.prev_label()
            
        @self.viewer.bind_key('PageUp', overwrite=True)
        def next_image_key(viewer):
            self.next_image()

        @self.viewer.bind_key("0", overwrite=True)
        def pan_switch0(viewer):
            self.pan()
            yield
            self.paint()
            
        @self.viewer.bind_key("Space", overwrite=True)
        def pan_switch1(viewer):
            self.pan()
            yield
            self.paint()
            
        @self.viewer.bind_key('Right', overwrite=True)
        def next_brush_size_key(viewer):
            self.next_brush_size() 
            time.sleep(150 / 1000) 
            self.next_brush_size_timer.start(10) 
            yield
            self.next_brush_size_timer.stop()
            
        @self.viewer.bind_key('Left', overwrite=True)
        def prev_brush_size_key(viewer):
            self.prev_brush_size() 
            time.sleep(150 / 1000) 
            self.prev_brush_size_timer.start(10) 
            yield
            self.prev_brush_size_timer.stop()
            
        @self.viewer.bind_key('Up', overwrite=True)
        def next_label_key(viewer):
            self.next_label()

        @self.viewer.bind_key('PageDown', overwrite=True)
        def previous_image_key(viewer):
            self.prev_image()
            
        @self.viewer.mouse_drag_callbacks.append
        def erase(viewer, event):
            if event.button==2:
                self.erase()
                yield
                self.paint()
                
#%% Function(s) general -------------------------------------------------------
                
    def next_image(self): 
        self.idx += 1
        self.update()
            
    def prev_image(self):
        self.idx -= 1
        self.update()
        
    def next_label(self):
        self.viewer.layers["mask"].selected_label += 1 

    def prev_label(self):
        if self.viewer.layers["mask"].selected_label > 1:
            self.viewer.layers["mask"].selected_label -= 1 
            
    def next_brush_size(self):
        self.viewer.layers["mask"].brush_size += 1
        
    def prev_brush_size(self):
        if self.viewer.layers["mask"].brush_size > 1:
            self.viewer.layers["mask"].brush_size -= 1
        
    def paint(self):
        self.viewer.layers["mask"].mode = 'paint'
            
    def erase(self):
        self.viewer.layers["mask"].mode = 'erase'
        
    def pan(self):
        self.viewer.layers["mask"].mode = 'pan_zoom'
        
#%% Function(s) open_image() --------------------------------------------------
        
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

#%% Function(s) save_mask() ---------------------------------------------------
       
    def save_mask(self):
        img_path = self.img_paths[self.idx]
        msk_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type}.tif"))
        io.imsave(msk_path, self.viewer.layers["mask"].data, check_contrast=False)  
        self.get_info_text()
       
#%% Function(s) get_info_text() -----------------------------------------------
        
    def get_info_text(self):
        
        def shorten_filename(name, max_length=32):
            if len(name) > max_length:
                parts = name.split('_')
                return parts[0] + "..." + parts[-1]
            else:
                return name
        
        img_path = self.img_paths[self.idx]
        msk_path = Path(
            str(img_path).replace(".tif", f"_mask-{mask_type}.tif"))
        img_name = img_path.name    
        if msk_path.exists() and edit:
            msk_name = msk_path.name 
        elif not msk_path.exists():
            msk_name = "None"

        # titles
        style0 = (
            " style='"
            "color: White;"
            "font-size: 10px;"
            "font-weight: normal;"
            "text-decoration: underline;"
            "'"
            )
        # filenames
        style1 = (
            " style='"
            "color: Khaki;"
            "font-size: 10px;"
            "font-weight: normal;"
            "text-decoration: none;"
            "'"
            ) 
        # Legend
        style2 = (
            " style='"
            "color: LightGray;"
            "font-size: 10px;"
            "font-weight: normal;"
            "text-decoration: none;"        
            "'"
            )
        # statistic values
        style3 = (
            " style='"
            "color: Tan;"
            "font-size: 10px;"
            "font-weight: normal;"
            "text-decoration: none;"
            "'"
            )
        # shortcut actions
        style4 = (
            " style='"
            "color: LightSteelBlue;"
            "font-size: 10px;"
            "font-weight: normal;"
            "text-decoration: none;"
            "'"
            )

        self.info.setText(
            
            # Image
            f"<p{style0}>Image<br><br>"
            f"<span {style1}>{shorten_filename(img_name, max_length=32)}</span>"
            
            # Mask
            f"<p{style0}>Mask<br><br>"
            f"<span{style1}>{shorten_filename(msk_name, max_length=32)}</span>"
            
            # Statistics
            f"<p{style0}>Statistics<br><br>"
            f"<span{style2}>- nObject(s)     {'&nbsp;' * 4}:</span>"
            f"<span{style3}> 0</span><br>"
            f"<span{style2}>- nLabel(s)      {'&nbsp;' * 5}:</span>"
            f"<span{style3}> 0</span><br>"
            f"<span{style2}>- minLabel       {'&nbsp;' * 6}:</span>"
            f"<span{style3}> 0</span><br>"
            f"<span{style2}>- maxLabel       {'&nbsp;' * 6}:</span>"
            f"<span{style3}> 0</span><br>"
            f"<span{style2}>- missLabel(s)   {'&nbsp;' * 2}:</span>"
            f"<span{style3}> 0</span><br>"            
            
            # Shortcuts
            f"<p{style0}>Shortcuts<br><br>"
            f"<span{style2}>- Save Mask      {'&nbsp;' * 5}:</span>"
            f"<span{style4}> End</span><br>"
            f"<span{style2}>- Next Image     {'&nbsp;' * 4}:</span>"
            f"<span{style4}> PageUp</span><br>"
            f"<span{style2}>- Previous Image {'&nbsp;' * 0}:</span>"
            f"<span{style4}> PageDown</span><br>"
            f"<span{style2}>- Next Label     {'&nbsp;' * 4}:</span>"
            f"<span{style4}> ArrowUp</span><br>"
            f"<span{style2}>- Previous Label {'&nbsp;' * 0}:</span>"
            f"<span{style4}> ArrowDown</span><br>"
            f"<span{style2}>- Increase brush {'&nbsp;' * 0}:</span>"
            f"<span{style4}> ArrowRight</span><br>"
            f"<span{style2}>- Decrease brush {'&nbsp;' * 0}:</span>"
            f"<span{style4}> ArrowLeft</span><br>"
            f"<span{style2}>- Paint tool     {'&nbsp;' * 4}:</span>"
            f"<span{style4}> LeftClick</span><br>"
            f"<span{style2}>- Erase tool     {'&nbsp;' * 4}:</span>"
            f"<span{style4}> RightClick</span><br>"
            f"<span{style2}>- Pan Image      {'&nbsp;' * 5}:</span>"
            f"<span{style4}> num[0]</span><br>"
            
            # f"<span{style2}>- Fill tool      {'&nbsp;' * 5}:</span>"
            # f"<span{style4}> Home</span><br>"
            
            )
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    painter = Painter(train_path, edit=edit, randomize=randomize)