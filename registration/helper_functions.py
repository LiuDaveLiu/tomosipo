import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# threshold images segmentation terms, low thr, high thr
def thr_image(image, thr_low, thr_high):
    window_image = image.copy()
    window_image[window_image < thr_low] = thr_low
    window_image[window_image > thr_high] = thr_high
    
    return window_image

def add_pad(image, height_pad=10, width_pad=10):  
    replicate = cv.copyMakeBorder(image,width_pad,width_pad,height_pad,height_pad,cv.BORDER_REPLICATE)    
    return replicate

def show_slice(slice):
   """
   Function to display an image slice
   Input is a numpy 2D array
   """
   plt.figure()
   plt.imshow(slice, cmap="gray")
