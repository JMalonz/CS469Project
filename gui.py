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
from tkinter import messagebox
from tkinter.filedialog import askdirectory
from PIL import Image, ImageFilter
from faceDetect import face_detection_filtering
import ntpath
import math
import numpy as np

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

        # Apply face detection (optional)
        if applyFaceDetection.get():
            originalImg = face_detection_filtering(file_path)
        
        clut_width, clut_height = haldImg.size
        clut_size = int(round(math.pow(clut_width, 1/3)))
        scalar = (clut_size * clut_size - 1) / 255
        originalImg = np.asarray(originalImg)
        # DO THIS PART IF WE HAVE ADDITIONAL TIME =========================================
        # This section intends to apply the CLUT if it is a grayscale one.
        # Need to restructure code to process b&w images instead of color.
        #if(len(np.asarray(haldImg).shape) < 3):
        #    haldImg = np.asarray(haldImg).reshape(clut_size ** 6)
        #    originalImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
        #else:
        #    haldImg = np.asarray(haldImg).reshape(clut_size ** 6, 3)
        # ===================================================================================
        
        haldImg = np.asarray(haldImg).reshape(clut_size ** 6, 3)
        # Correspond 3D CLUT indices to corresponding pixels in the image
        # clut_red, clut_green, and clut_blue all together represent 1 cell in the 3D clut.
        # use rint here instead of round because we need integers, not decimals.
        clut_red = np.rint(originalImg[:, :, 0] * scalar).astype(int)
        clut_green = np.rint(originalImg[:, :, 1] * scalar).astype(int)
        clut_blue = np.rint(originalImg[:, :, 2] * scalar).astype(int)

        # Initialize a zeroed out array in the same resolution as the original image
        modified_img = np.zeros((originalImg.shape))

        # For grain/noise generation
        noise = np.random.normal(500, 200, modified_img.shape)
        
        # Apply coloring mapping
        modified_img[:, :] = haldImg[clut_red + clut_size ** 2 * clut_green + clut_size ** 4 * clut_blue]
        # Convert to image from array so that we can display 
        modified_img = Image.fromarray(modified_img.astype('uint8'), 'RGB')
        # Convert noise to image from array so that we can blend
        noise = Image.fromarray(noise.astype('uint8'), 'RGB')
        # 0.15 alpha chosen to have the noise be mostly transparent but still noticeable
        modified_img = Image.blend(modified_img, noise, 0.15)
        # Median Filter to make noise look more believeable as film
        modified_img = modified_img.filter(ImageFilter.MedianFilter(size = 3))
        file = filedialog.asksaveasfile(mode = 'w', defaultextension = ".png", filetypes=[("PNG", "*.png;*.PNG"), ("JPEG", "*.jpg;*.jpeg;*.JPG;*.JPEG"), ("All Files", "*.*")])
        if file:
            save_path = ntpath.abspath(file.name)
            modified_img.save(save_path)
            messagebox.showinfo("Success!", message="Image has been saved!")
        #modified_img.show()
        #noise.show()

def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("PNG or JPEG", "*.png;*.jpg;*.jpeg")])
    if file_path:
        ttk.Label(mainframe, text = "Selected Image: " + ntpath.basename(file_path)).grid(column=1, row=2, sticky=W)
        

def open_hald_dialog():
    global hald_file_path
    hald_file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("PNG or JPEG", "*.png;*.jpg;*.jpeg")])
    if hald_file_path:
        ttk.Label(mainframe, text = "Selected CLUT: " + ntpath.basename(hald_file_path)).grid(column=1, row=3, sticky=W)
        
#def chooseSaveDir():
#    global save_path
#    save_path = askdirectory(title='Select a folder')

root = Tk()
root.geometry("600x250")
root.title("Film Simulation Program")

mainframe = ttk.Frame(root, padding="1 1 1 1")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S, NW, NE, SE, SW))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(mainframe, text="Please select the image you wish to apply the filter to.\nThen select the clut/filter you wish to apply.").grid(column=1, row=1, sticky=NW)

openImageButton = ttk.Button(mainframe, text="Open Color Image", command = open_file_dialog)
openHaldButton = ttk.Button(mainframe, text="Open Color HaldCLUT", command = open_hald_dialog)
#saveDirButton = ttk.Button(mainframe, text="Choose Save Folder", command = chooseSaveDir)
applyButton = ttk.Button(mainframe, text="Apply", command=apply_HaldClut)
global applyFaceDetection
applyFaceDetection = IntVar(root)
faceDetectCheckbox = ttk.Checkbutton(mainframe, text="Apply face detection and filtering", variable = applyFaceDetection)

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

root.mainloop()
