#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# Qt
from qtpy.QtGui import QFont
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QLabel, QFrame

# Skimage
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries, expand_labels

#%% Comments ------------------------------------------------------------------

'''
- Adjust statistic display (color, alignement...)
- Better manage mask_type variable (maybe available from interface?)
- B&C, auto? in the interface?
- Fill objects (auto? shortcut? both?) Ongoing (Ctrl + right click to erase full label)
- better tune brush resize method
- Reset view on first image?
'''

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd().parent, "data", "train_spores") 

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
        self.edit = edit
        self.randomize = randomize
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
        self.open_image()
        self.get_info_text()
        
#%% Initialize ----------------------------------------------------------------
        
    def init_paths(self):
        self.img_paths, self.msk_paths = [], []
        for img_path in self.train_path.iterdir():
            if "mask" not in img_path.name:
                self.img_paths.append(img_path)
                self.msk_paths.append(
                    Path(str(img_path).replace(".tif", f"_mask-{mask_type}.tif"))
                    )
        if self.randomize:
            permutation = np.random.permutation(len(self.img_paths))
            self.img_paths = [self.img_paths[i] for i in permutation]
            self.msk_paths = [self.msk_paths[i] for i in permutation]
            
    def init_images(self):
        self.imgs, self.msks = [], []
        for img_path, msk_path in zip(self.img_paths, self.msk_paths):
            img = io.imread(img_path)
            if msk_path.exists() and self.edit:   
                msk = io.imread(msk_path)
            elif not msk_path.exists():
                msk = np.zeros_like(img, dtype="uint8")
            self.imgs.append(img)
            self.msks.append(msk)
        
    def init_viewer(self):
               
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.imgs[0], name="image")
        self.viewer.add_labels(self.msks[0], name="mask")

        # Create widget
        self.widget = QWidget()
        self.layout = QVBoxLayout()

        # Create buttons
        btn_next_image = QPushButton("Next Image")
        btn_prev_image = QPushButton("Previous Image")
        btn_save_mask = QPushButton("Save Mask")
        btn_solve_labels = QPushButton("Solve Labels")

        # Create texts
        self.info0  = QLabel()
        self.info0.setFont(QFont("Consolas"))
        self.info1  = QLabel()
        self.info1.setFont(QFont("Consolas"))
        self.info2  = QLabel()
        self.info2.setFont(QFont("Consolas"))
        
        # Add buttons and text to layout
        
        self.layout.addWidget(btn_next_image)
        self.layout.addWidget(btn_prev_image)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info0)
        self.layout.addSpacing(10)
        
        self.layout.addWidget(btn_solve_labels)
        self.layout.addWidget(btn_save_mask)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info1)
        self.layout.addSpacing(10)
        
        self.layout.addWidget(self.info2)

        # Add layout to widget
        self.widget.setLayout(self.layout)

        # Connect buttons
        btn_next_image.clicked.connect(self.next_image)
        btn_prev_image.clicked.connect(self.prev_image)
        btn_save_mask.clicked.connect(self.save_mask)
        btn_solve_labels.clicked.connect(self.solve_labels)

        # Add the widget to viewer
        self.viewer.window.add_dock_widget(
            self.widget, area='right', name="Painter")
                
#%% Shortcuts -----------------------------------------------------------------
        
        # Viewer

        @self.viewer.bind_key("End", overwrite=True)
        def save_mask_key(viewer):
            self.save_mask() 
        
        @self.viewer.bind_key('PageDown', overwrite=True)
        def previous_image_key(viewer):
            self.prev_image()
        
        @self.viewer.bind_key('PageUp', overwrite=True)
        def next_image_key(viewer):
            self.next_image()
            
        @self.viewer.bind_key("0", overwrite=True)
        def pan_switch_key0(viewer):
            self.pan()
            yield
            self.paint()
            
        @self.viewer.bind_key("Space", overwrite=True)
        def pan_switch_key1(viewer):
            self.pan()
            yield
            self.paint()
            
        @self.viewer.bind_key('Backspace', overwrite=True)
        def hide_labels_key(viewer):
            self.hide_labels()
            yield
            self.show_labels()
            
        @self.viewer.bind_key('Home', overwrite=True)
        def reset_view_key(viewer):
            self.reset_view()
            
        # Paint
            
        @self.viewer.bind_key('Down', overwrite=True)
        def prev_label_key(viewer):
            self.prev_label()
            
        @self.viewer.bind_key('Up', overwrite=True)
        def next_label_key(viewer):
            self.next_label()
            
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
            
        @self.viewer.mouse_drag_callbacks.append
        def erase(viewer, event):
            
            if event.button == 2:
                self.erase()
                yield
                self.paint()
            
            if 'Control' in event.modifiers:
                if event.button == 1:
                    self.fill()
                    yield
                    self.paint()
                    
            if 'Control' in event.modifiers:
                if event.button == 2:
                    self.fill()
                    yield
                    self.paint()
                
#%% Function(s) shortcuts -----------------------------------------------------
                
    # Viewer    

    def prev_image(self):
        self.idx -= 1
        self.update()
        
    def next_image(self): 
        self.idx += 1
        self.update()
        
    def pan(self):
        self.viewer.layers["mask"].mode = 'pan_zoom'
        
    def show_labels(self):
        self.viewer.layers["mask"].visible = True
    
    def hide_labels(self):
        self.viewer.layers["mask"].visible = False  
        
    def reset_view(self):
        self.viewer.reset_view()

    # Paint
    
    def prev_label(self):
        if self.viewer.layers["mask"].selected_label > 1:
            self.viewer.layers["mask"].selected_label -= 1 
            
    def next_label(self):
        self.viewer.layers["mask"].selected_label += 1 
        
    def prev_brush_size(self):
        if self.viewer.layers["mask"].brush_size > 1:
            self.viewer.layers["mask"].brush_size -= 1
        
    def next_brush_size(self):
        self.viewer.layers["mask"].brush_size += 1

    def paint(self):
        self.viewer.layers["mask"].mode = 'paint'
            
    def erase(self):
        self.viewer.layers["mask"].mode = 'erase'
        
    def fill(self):
        self.viewer.layers["mask"].mode = "fill"

    # def erase_label(self):
        
        

#%% Function(s) open_image() --------------------------------------------------
        
    def open_image(self):
        self.viewer.layers["image"].data = self.imgs[self.idx]
        self.viewer.layers["mask"].data = self.msks[self.idx]
        self.viewer.layers["image"].contrast_limits = contrast_limits
        self.viewer.layers["image"].gamma = 0.66
        self.viewer.layers["mask"].brush_size = brush_size
        self.viewer.layers["mask"].selected_label = 1
        self.viewer.layers["mask"].mode = 'paint'
        self.reset_view()

#%% Function(s) get_stats() ---------------------------------------------------
       
    def get_stats(self):
        
        msk = self.viewer.layers["mask"].data
        msk_obj = label(msk > 0 ^ find_boundaries(msk), connectivity=1)
        self.nObjects = np.maximum(0, len(np.unique(msk_obj)) - 1)
        self.nLabels = np.maximum(0, len(np.unique(msk)) - 1)
        self.minLabel = np.min(msk)
        self.maxLabel = np.max(msk)

        # Get missing labels (between min & max label)
        self.missLabels = []
        for lbl in range(self.maxLabel):
            if np.all(msk != lbl):
                self.missLabels.append(f"{lbl}")
        self.missLabels = ", ".join(self.missLabels) 

        # Get duplicated labels (multi objects label)
        lbls = []
        for props in regionprops(msk_obj, intensity_image=msk):
            lbls.append(int(props.intensity_max))
        uniq, count = np.unique(lbls, return_counts=True)
        self.dupLabels = []
        for l, lbl in enumerate(uniq):
            if count[l] > 1:
                self.dupLabels.append(f"{lbl}({count[l]})")
        self.dupLabels = ", ".join(self.dupLabels)
        
#%% Function(s) solve_labels() ------------------------------------------------

    def solve_labels(self):
        msk = self.viewer.layers["mask"].data
        msk_obj = msk.copy()
        msk_obj[find_boundaries(msk) == 1] = 0
        msk_obj = label(msk_obj, connectivity=1)
        msk_obj = expand_labels(msk_obj)
        msk_obj[msk == 0] = 0
        self.viewer.layers["mask"].data = msk_obj
        self.get_info_text()
            
#%% Function(s) save_mask() ---------------------------------------------------
       
    def save_mask(self):
        msk = self.viewer.layers["mask"].data
        msk_path = self.msk_paths[self.idx]
        self.msks[self.idx] = msk
        io.imsave(msk_path, msk, check_contrast=False) 
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
        msk_path = self.msk_paths[self.idx]
        img_name = img_path.name    
        if msk_path.exists() and edit:
            msk_name = msk_path.name 
        elif not msk_path.exists():
            msk_name = "None"
            
        self.get_stats()

        # titles
        style0 = (
            " style='"
            "color: White;"
            "font-size: 12px;"
            "font-weight: normal;"
            "text-decoration: underline;"
            "'"
            )
        # filenames
        style1 = (
            " style='"
            "color: Khaki;"
            "font-size: 12px;"
            "font-weight: normal;"
            "text-decoration: none;"
            "'"
            ) 
        # Legend
        style2 = (
            " style='"
            "color: LightGray;"
            "font-size: 12px;"
            "font-weight: normal;"
            "text-decoration: none;"        
            "'"
            )
        # statistic values
        style3 = (
            " style='"
            "color: Brown;"
            "font-size: 12px;"
            "font-weight: normal;"
            "text-decoration: none;"
            "'"
            )
        # shortcut actions
        style4 = (
            " style='"
            "color: LightSteelBlue;"
            "font-size: 12px;"
            "font-weight: normal;"
            "text-decoration: none;"
            "'"
            )

        self.info0.setText(
            
            # Image
            f"<p{style0}>Image<br><br>"
            f"<span {style1}>{shorten_filename(img_name, max_length=32)}</span>"
            
            # Mask
            f"<p{style0}>Mask<br><br>"
            f"<span{style1}>{shorten_filename(msk_name, max_length=32)}</span>"
            
            )
            
        self.info1.setText(
            
            # Statistics
            f"<p{style0}>Statistics<br><br>"
            f"<span{style2}>- nObject(s)     {'&nbsp;' * 4}:</span>"
            f"<span{style3}> {self.nObjects}</span><br>"
            f"<span{style2}>- nLabel(s)      {'&nbsp;' * 5}:</span>"
            f"<span{style3}> {self.nLabels}</span><br>"
            f"<span{style2}>- minLabel       {'&nbsp;' * 6}:</span>"
            f"<span{style3}> {self.minLabel}</span><br>"
            f"<span{style2}>- maxLabel       {'&nbsp;' * 6}:</span>"
            f"<span{style3}> {self.maxLabel}</span><br>"
            f"<span{style2}>- missLabel(s)   {'&nbsp;' * 2}:</span>"
            f"<span{style3}> {self.missLabels}</span><br>"       
            f"<span{style2}>- dupLabel(s)    {'&nbsp;' * 3}:</span>"
            f"<span{style3}> {self.dupLabels}</span>" 
            
            )
        
        self.info2.setText(
        
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
            f"<span{style2}>- Hide Labels    {'&nbsp;' * 3}:</span>"
            f"<span{style4}> Backspace</span><br>"
            f"<span{style2}>- Increase brush {'&nbsp;' * 0}:</span>"
            f"<span{style4}> ArrowRight</span><br>"
            f"<span{style2}>- Decrease brush {'&nbsp;' * 0}:</span>"
            f"<span{style4}> ArrowLeft</span><br>"
            f"<span{style2}>- Paint tool     {'&nbsp;' * 4}:</span>"
            f"<span{style4}> Mouse[left]</span><br>"
            f"<span{style2}>- Erase tool     {'&nbsp;' * 4}:</span>"
            f"<span{style4}> Mouse[right]</span><br>"
            f"<span{style2}>- Pan Image      {'&nbsp;' * 5}:</span>"
            f"<span{style4}> num[0]</span><br>"
            
            # f"<span{style2}>- Fill tool      {'&nbsp;' * 5}:</span>"
            # f"<span{style4}> Home</span><br>"
            
            )
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    painter = Painter(train_path, edit=edit, randomize=randomize)