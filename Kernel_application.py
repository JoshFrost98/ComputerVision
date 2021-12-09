import numpy as np
import pandas as pd
import kernel as k
import cv2
import math



def ICV_import_image_rgb(name): #imports image
    image = cv2.imread(name, cv2.IMREAD_COLOR)
    blue = image[:,:,0]
    green = image[:,:,1]
    red = image[:,:,2]
    return red, green, blue

def ICV_to_grey(r, g, b): #changes to greyscale
    image = 0.2989*r + 0.5870*g + 0.1140*b
    return image

def ICV_run():
    r, g, b = ICV_import_image_rgb('Josh_Frost_Profile.png') #import images
    image = ICV_to_grey(r, g, b)

    # INITIALIZE KERNELS
    intensity =[
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ]
    edges = [
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ]
    gaussian = [
        [1/16,2/16,1/16],
        [2/16,4/16,1/16],
        [1/16,2/16,1/16]
    ]
    bad_gaussian = [
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ]
    vert = [
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
    ]
    emboss = [
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ]
    #also a kernel
    hori = np.transpose(vert)
    
    image1 =  k.ICV_apply_kernel(image, edges)#applies edges kernel to image using kernel function from kernel.py
    image1 = image1.astype(np.uint8)#changes values so it can be shown using imshow
    
    
    cv2.imshow('changed face', image1.astype(np.uint8))
    cv2.waitKey(0)


ICV_run()