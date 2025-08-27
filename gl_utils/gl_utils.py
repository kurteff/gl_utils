import numpy as np
import os
from PIL import Image
from colorsys import rgb_to_hsv


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