# File name: CLUT.py
# Description: The purpose of this file will be to apply a CLUT to an image.
# A CLUT is a Color Look Up Table. In other words, we are mapping colors to other colors.
# We do this because film stocks tend to have a "color profile". These profiles tend to
# lean heavily into certain colors and mute others. For example, the Fuji Velvia 50 film stock.
# This film stock tends to favor red tones so reds will appear darker than their digital counter parts.
# Oranges also tend to look a bit darker and have more red tones in them.

import imageio as iio

# Initial image testing
img = iio.imread("TestPictures/iPhone6sPlus_SingleFace.JPG")
