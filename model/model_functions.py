#%% Imports -------------------------------------------------------------------

from pathlib import Path

#%% Function : update_info_text() ---------------------------------------------

def update_info_text(img_path, mask_type):
    
    def shorten_filename(name, max_length=32):
        if len(name) > max_length:
            parts = name.split('_')
            return parts[0] + "..." + parts[-1]
        else:
            return name
    
    msk_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type}.tif"))
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
    # shortcuts (name)
    style2 = (
        " style='"
        "color: LightGray;"
        "font-size: 10px;"
        "font-weight: normal;"
        "text-decoration: none;"        
        "'"
        )
    # shortcuts (action)
    style3 = (
        " style='"
        "color: LightSteelBlue;"
        "font-size: 10px;"
        "font-weight: normal;"
        "text-decoration: none;"
        "'"
        )

    info_text = (
        
        # Image
        f"<p{style0}>Image<br>"
        f"<span {style1}>{shorten_filename(img_name, max_length=32)}</span>"
        
        # Mask
        f"<p{style0}>Mask<br>"
        f"<span{style1}>{shorten_filename(msk_name, max_length=32)}</span>"
        
        # Shortcuts
        f"<p{style0}>Shortcuts<br>"
        f"<span{style2}>- Save Mask      {'&nbsp;' * 5}:</span>"
        f"<span{style3}> Enter</span><br>"
        f"<span{style2}>- Next Image     {'&nbsp;' * 4}:</span>"
        f"<span{style3}> PageUp</span><br>"
        f"<span{style2}>- Previous Image {'&nbsp;' * 0}:</span>"
        f"<span{style3}> PageDown</span><br>"
        f"<span{style2}>- Next Label     {'&nbsp;' * 4}:</span>"
        f"<span{style3}> ArrowUp</span><br>"
        f"<span{style2}>- Previous Label {'&nbsp;' * 0}:</span>"
        f"<span{style3}> ArrowDown</span><br>"
        f"<span{style2}>- Increase brush {'&nbsp;' * 0}:</span>"
        f"<span{style3}> ArrowRight</span><br>"
        f"<span{style2}>- Decrease brush {'&nbsp;' * 0}:</span>"
        f"<span{style3}> ArrowLeft</span><br>"
        f"<span{style2}>- Erase tool     {'&nbsp;' * 4}:</span>"
        f"<span{style3}> End</span><br>"
        f"<span{style2}>- Fill tool      {'&nbsp;' * 5}:</span>"
        f"<span{style3}> Home</span><br>"
        f"<span{style2}>- Pan Image      {'&nbsp;' * 5}:</span>"
        f"<span{style3}> Space</span><br>"
        
        )
    
    return info_text
