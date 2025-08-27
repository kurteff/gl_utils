# RUN THIS SCRIPT INSIDE MRICROGL!

# explode_ensemble.py
# Written by Lynn Kurteff for the Hickok Lab
# August 25, 2025

# Imports
import gl
import time
import os
print(time.time())

### File paths - CHANGE THESE LOCALLY -->
# Overlay
overlay_path = '/path/to/overlay.nii.gz'
# Background image, can be a file path, or the name of a standard
# brain that ships with MRIcroGL
image_path = 'mni152'
# Where to save the screenshots
img_dir = '/path/to/screenshot_folder/'
### <--

### Parameters - update these as you see fit -->
wtime = 100 # msec to wait between drawing slices

# Where to position slices (has to be ascending)
# These are percent values in the coronal plane; e.g., if you're loading
# an ACPC aligned image then depth 0.5 is on the AC.
depths = [0, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1]

# Camera controls (default values here are good for LH)
azi = 130
elev = 20

show_colorbar = 0 # 0=off, 1=top, 2=right

# name of colormap in MRIcroGL
overlay_color = 'viridis'

# Background color (RGB tuple)
# Default is an ugly green (we are going to 'green screen' this out in step 2)
bgcolor = (0,255,0)
### <--

gl.resetdefaults()
gl.colorbarposition(show_colorbar)
gl.bmptransparent(0) # setting this to 1 fucks your colors up

# Template
gl.loadimage(image_path)
gl.opacity(0, 100)

# Overlay (ensemble)
# Change layer to 2 if also using candidate
gl.overlayload(overlay_path)
gl.colorname(1, overlay_color)
gl.minmax(1, 0, 1)
gl.opacity(1, 100)

# Shader configs
gl.shaderadjust('boundThresh', 0.35)
gl.shaderadjust('edgeThresh', 0.42)
gl.shaderadjust('edgeBoundMix',0.05)
gl.shaderadjust('colorTemp', 0.8)
gl.shaderadjust('overlayClip', 1)

# Camera
gl.azimuthelevation(azi,elev)

# Green screen
r,g,b = bgcolor
gl.backcolor(r,g,b)

# Loop/plot
pad = 0.001
for i,depth in enumerate(depths):
    if i != len(depths)-1: # Don't do the last slice
        thick = depths[i+1] - depth
        if i == 0: # i dont know why i have to do this
            thick = thick/2
        depth += pad

        gl.clipthick(thick)
        gl.clipazimuthelevation(depth, 0, 180);
        gl.wait(wtime)

        gl.savebmp(os.path.join(img_dir, f'slice_{i}.png'))
