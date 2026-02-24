import numpy as np
import os
from PIL import Image
from glob import glob
from colorsys import rgb_to_hsv
from matplotlib import pyplot as plt
import nibabel as nib # DICOM/NIFTI imaging support
from nibabel.processing import resample_from_to # reslicing DICOMs
from nilearn import datasets # anatomical atlases
from nilearn.image import coord_transform # NIFTI manipulation

# # # # # # # # # # # # # # # #
#        NIFTI creation       #
# # # # # # # # # # # # # # # #
def nearest(t,a):
    '''
    Returns index and value from array a that is closest to value t.
    '''
    i = np.abs(a-t).argmin()
    v = a[i]
    return i,v
    
def create_bin_sphere(arr_size, center, r):
    '''
    Creates a binary mask for a sphere with a given radius at a given midpoint.
    See: gl_utils.draw_mni_sphere()
    '''
    coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
    distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
    return 1*(distance <= r)

def draw_mni_sphere(mni_xyz, arr_size, radius, affine, coordsystem='ras', mask_by=False, mask_atlas='harvard-oxford',
    mask_roi="Precentral Gyrus", p_thresh=20, fname=None, debug=False):
    '''
    Given a point in voxel space, an affine transformation, and a millimeter radius,
    draws a sphere around that point in world (MNI) space.
    Returns the sphere as a nibabel image.
    * mni_xyz: tuple of (x,y,z) of the sphere's center in MNI coordinates.
    * arr_size: tuple of shape of your array of voxels (x, y, z).
    * radius: in mm (float or int OK)
    * affine: transformation from voxel to world coordinates.
    * coordsystem: How your voxel array is oriented. Currently, only RAS is supported.
    * mask_by: if True, clips voxels outside a specified atlas.
        * mask_atlas: name of atlas you want to mask by. Currently, only 'harvard-oxford' supported.
        * mask_roi: ROI name within atlas you're masking by. See also: utils.print_atlas_labels()
        * p_thresh: Percent (0-100) probability that a voxel is in an atlas's ROI for thresholding voxels within ROI.
    * fname: saves the nibabel image as a NIFTI file using the specified fname, unless None, then does not save file.
    * debug: bool, controls print statements for debugging.

    TO DO:
        * add support for non-RAS coordsystems
        * add support for clipping to other atlases (pass as arg)
    '''
    if coordsystem != 'ras':
        raise Exception("Coordinate systems other than RAS have yet to be implemented. Please convert your array to RAS.")

    if len(np.unique(np.diag(affine)[:-1])) != 1:
        raise Exception("Non-cubic voxels are currently unsupported. Please check your affine.")

    # get column, row, slice of specified MNI coordinate in the voxel array
    x, y, z = mni_xyz
    c, _ = nearest(x, np.array([coord_transform(r,0,0,affine)[0] for r in np.arange(arr_size[0])]))
    r, _ = nearest(y, np.array([coord_transform(0,a,0,affine)[1] for a in np.arange(arr_size[1])]))
    s, _ = nearest(z, np.array([coord_transform(0,0,s,affine)[2] for s in np.arange(arr_size[2])]))
    crs = [c,r,s]

    # convert voxels to mm
    vx_mm_ratio = np.diag(affine)[0]
    radius_vx = np.ceil(radius/vx_mm_ratio)

    sphere = create_bin_sphere(arr_size, crs, radius_vx).astype(float)

    if mask_by:
        if mask_atlas not in ['harvard-oxford']:
            raise Exception(f"Unsupported atlas {mask_atlas}.")
        atlas = datasets.fetch_atlas_harvard_oxford("cort-prob-1mm")
        if mask_roi not in atlas['labels']:
            raise Exception(f"ROI {mask_roi} not found in {mask_atlas} atlas. Please double-check atlas labels using utils.print_atlas_labels().")
        # Extract ROI, convert to 3D   
        atlas_4d = atlas['maps'].get_fdata()
        roi_idx = atlas['labels'].index(mask_roi) - 1 # -1 because of "Background"
        atlas_thresh = (atlas_4d[:,:,:,roi_idx] > p_thresh).astype(float)
        roi = nib.Nifti1Image(atlas_thresh, atlas['maps'].affine)
        # Resize to match voxel space's dimensions and binarize to make a mask
        roi_zoom = resample_from_to(roi, (arr_size, affine))
        roi_resp = roi_zoom.get_fdata()
        mask = (roi_resp > 0.75).astype(float) # 0.75 was arbitrated based on looking at histograms of resliced ROIs
        # Clip the sphere (sorry for spaghetti code)
        sphere_clipped = np.zeros(sphere.shape)
        cols, rows, slices = sphere.shape
        for c in np.arange(cols):
            for r in np.arange(rows):
                for s in np.arange(slices):
                    if sphere[c,r,s] == 1 and mask[c,r,s] == 1:
                        sphere_clipped[c,r,s] = 1
        if debug:
            nvoxels = sum(sphere.ravel())
            nvoxels_clipped = sum(sphere_clipped.ravel())
            print(f"Orig sphere contained {nvoxels} voxels; clipped sphere contains {nvoxels_clipped} (%.2f%%)"%(nvoxels_clipped/nvoxels))
        # Overwrite orig sphere with clipped one
        sphere = sphere_clipped

    # Return sphere (and save if specified in args)
    nifti = nib.Nifti1Image(sphere, affine)
    if fname is not None:
        nib.save(nifti, fname)
    return nifti


# # # # # # # # # # # # # # # #
# Screenshot post-processing  #
# # # # # # # # # # # # # # # #
def greenscreen(image_path, chromakey=(0, 255, 0), tolerance=0, method='euclidean'):
    """
    Convert green screen background to transparent pixels.
    
    Args:
        image_path (str): Path to the input image
        chromakey (tuple): RGB values of the green screen color (default: pure green)
        tolerance (float): Tolerance for color matching
    
    Returns:
        numpy.ndarray: RGBA image array with transparent background
    """
    img = Image.open(image_path)
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
    rgba_array[:, :, :3] = img_array  # Copy RGB channels
    rgba_array[:, :, 3] = 255  # Set alpha to fully opaque initially

    # Professional chroma key algorithm
    r, g, b = img_array[:, :, 0].astype(float), img_array[:, :, 1].astype(float), img_array[:, :, 2].astype(float)
    green_key = g - np.maximum(r, b)
    max_green_key = np.max(green_key)
    if max_green_key > 0:
        green_key_normalized = green_key / max_green_key
        ckey_mask = green_key_normalized > (1 - tolerance)
    else:
        ckey_mask = np.zeros(img_array.shape[:2], dtype=bool)

    rgba_array[ckey_mask, 3] = 0
    
    return rgba_array

def explode(imgdir, xpad=450, ypad=300, title=False, showfig=False):
    '''
    Greenscreens and concatenates images to make an AIO 'exploded' plot
    in the style of Utianski et al. 2018.

    Parameters
    --------------
    imgdir : str
        Path to folder containing slices you wish to concatenate
    xpad : int
        Padding (in pixels) to space images on x axis
    ypad : int
        xpad, but for the y axis, duh
    title : bool or str, default False
        Whether or not to title the image. If str, uses that as title.
    showfig : bool
        Whether or not to leave the figure open after saving to file
    '''
    images = glob(imgdir + '/*slice*')
    width, height = Image.open(images[0]).size
    total_width = width + (len(images) - 1) * xpad
    total_height = height + (len(images) -1) * ypad
    aspect_ratio = total_height / total_width
    figsize = (int(total_width/ypad), int(total_height/ypad))
    plt.figure(figsize=figsize)
    plt.gca().set_xlim(xpad, total_width-(xpad/2))
    plt.gca().set_ylim(ypad*2, total_height-(ypad*2))
    for i,image_path in enumerate(images):
        img = greenscreen(image_path, chromakey=(0,255,0), tolerance=1, method='chroma_key')
        # Delete cube
        img[-250:,:250] = 0.
        if i == 0: # Spaghetti code, I don't know why this works but it does ok
            x_offset = xpad/3
            y_offset = ypad/3
        else:
            x_offset = i * xpad
            y_offset = i * ypad
        plt.imshow(img, extent=[x_offset, x_offset + width, y_offset, y_offset + height], zorder=len(images)-i)
    plt.axis('off');
    if title:
        plt.title(title, fontsize=32)
    plt.savefig(os.path.join(imgdir, f'explode.png'))
    if not showfig:
        plt.close();