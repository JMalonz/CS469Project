# File name: gui.py
# Description: The purpose of this file will be to apply a CLUT to an image.
# A CLUT is a Color Look Up Table. In other words, we are mapping colors to other colors.
# We do this because film stocks tend to have a "color profile". These profiles tend to
# lean heavily into certain colors and mute others. For example, the Fuji Velvia 50 film stock.
# This film stock tends to favor red tones so reds will appear darker than their digital counter parts.
# Oranges also tend to look a bit darker and have more red tones in them.

# This file also contains the GUI.

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
import ntpath
import math
import numpy as np
import cv2

#imgTest = Image.open("Test Pictures/iPhone6sPlus_SingleFace.JPG")
#haldTest = Image.open("CLUTs/HaldCLUT/Color/Lomography/Lomography X-Pro Slide 200.png")

# apply_HaldClut will use the globally accessible file_path and hald_file_path.
# Previously, it would take a 
# In effect, we get the value of a pixel and map it to a color present in the CLUT which is represented by a 3D cube.
# Once mapped, replace the original pixel color with the color from the CLUT.
def apply_HaldClut():
    if(not file_path or not hald_file_path):
        ttk.Label(mainframe, text = "Please select a HALD CLUT and image file before pressing apply.").grid(column=1, row=3, sticky=W)
        return -1
    else:
        # Open Images
        haldImg = Image.open(hald_file_path)
        originalImg = Image.open(file_path)
        
        clut_width, clut_height = haldImg.size
        clut_size = int(round(math.pow(clut_width, 1/3)))
        scalar = (clut_size * clut_size - 1) / 255
        originalImg = np.asarray(originalImg)
        haldImg = np.asarray(haldImg).reshape(clut_size ** 6, 3)

        # Correspond 3D CLUT indices to corresponding pixels in the image
        # clut_red, clut_green, and clut_blue all together represent 1 cell in the 3D clut.
        # use rint here instead of round because we need integers, not decimals.
        # 
        clut_red = np.rint(originalImg[:, :, 0] * scalar).astype(int)
        clut_green = np.rint(originalImg[:, :, 1] * scalar).astype(int)
        clut_blue = np.rint(originalImg[:, :, 2] * scalar).astype(int)

        # Initialize a zeroed out array in the same resolution as the original image
        modified_img = np.zeros((originalImg.shape))

        # For grain/noise generation, create an empty array the same size as the original image
        noise = np.zeros((originalImg.shape), dtype = np.uint8)
        cv2.randn(noise, 400, 50)
        noise = (noise * 0.5).astype(np.uint8)
        
        # Add noise to image
        
        # Apply coloring mapping
        modified_img[:, :] = haldImg[clut_red + clut_size ** 2 * clut_green + clut_size ** 4 * clut_blue] 
        modified_img = Image.fromarray(modified_img.astype('uint8'), 'RGB')
        modified_img.show()
        noise = Image.fromarray(noise.astype('uint8'), 'RGB')
        noise.show()

def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("PNG or JPEG", "*.png;*.jpg;*.jpeg")])
    if file_path:
        ttk.Label(mainframe, text = "Selected Image: " + ntpath.basename(file_path)).grid(column=3, row=2, sticky=W)
        

def open_hald_dialog():
    global hald_file_path
    hald_file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("PNG or JPEG", "*.png;*.jpg;*.jpeg")])
    if hald_file_path:
        ttk.Label(mainframe, text = "Selected CLUT: " + ntpath.basename(hald_file_path)).grid(column=3, row=3, sticky=W)
        

root = Tk()
root.geometry("600x250")
root.title("Film Simulation Program")

mainframe = ttk.Frame(root, padding="1 1 1 1")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S, NW, NE, SE, SW))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(mainframe, text="Please select the image you wish to apply the filter to.\nThen select the clut/filter you wish to apply.").grid(column=1, row=1, sticky=NW)

openImageButton = ttk.Button(mainframe, text="Open Image", command = open_file_dialog)
openHaldButton = ttk.Button(mainframe, text="Open HaldCLUT", command = open_hald_dialog)
applyButton = ttk.Button(mainframe, text="Apply", command=apply_HaldClut)

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

root.mainloop()
