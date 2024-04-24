# File name: CLUT.py
# Description: The purpose of this file will be to apply a CLUT to an image.
# A CLUT is a Color Look Up Table. In other words, we are mapping colors to other colors.
# We do this because film stocks tend to have a "color profile". These profiles tend to
# lean heavily into certain colors and mute others. For example, the Fuji Velvia 50 film stock.
# This film stock tends to favor red tones so reds will appear darker than their digital counter parts.
# Oranges also tend to look a bit darker and have more red tones in them.


import numpy as np
from PIL import Image
import math

def apply_clut(clut, img):
    clut_width, clut_height = clut.size
    clut_size = int(round(math.pow(clut_width, 1/3)))
    # Take cube root of clut_size because 12-bit CLUT has the same information as 144-bit 3D clut
    scalar = (clut_size * clut_size - 1) / 255
    img = np.asarray(img)
    clut = np.asarray(clut).reshape(clut_size ** 6, 3)

    # Correspond 3D CLUT indices to corresponding pixels in the image
    clut_red = np.rint(img[:, :, 0] * scalar).astype(int)
    clut_green = np.rint(img[:, :, 1] * scalar).astype(int)
    clut_blue = np.rint(img[:, :, 2] * scalar).astype(int)

    modified_img = np.zeros((img.shape))

    modified_img[:, :] = clut[clut_red + clut_size ** 2 * clut_green + clut_size ** 4 * clut_blue]
    modified_img = Image.fromarray(modified_img.astype('uint8'), 'RGB')
    return modified_img

# Initial image testing
imgTest = Image.open("Test Pictures/iPhone6sPlus_SingleFace.JPG")
haldTest = Image.open("CLUTs/HaldCLUT/Color/Lomography/Lomography Redscale 100.png")
moddedImg = apply_clut(haldTest, imgTest)
moddedImg.show()