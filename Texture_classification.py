import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


def ICV_fill_border(image):#fills border values in an array, using the values from their neighbors.
    image = np.array(image)
    x =len(image[:,0])
    y =len(image[0])
    image[0,:] = image[1,:]
    image[x-1,:] = image[x-2,:]
    image[:,0] = image[:,1]
    image[:,y-1] = image[:,y-2]
    return image

def ICV_plot(hist, no):#plots vals
    plt.figure(no)
    plt.plot(hist)

def ICV_split_image(image, windowsvert,windowslat): # splits image into equal sized windows, takes a greyscale image, would have to be called for each channel in RGB
    shape = image.shape
    if shape[0] % windowsvert != 0:#if window size doesnt fit, then a section of the image is lost, and only what can be split into windows is sent back
        image = image[0:(shape[0]-shape[0]%windowsvert),:]
    elif shape[1] % windowslat != 0:
        image = image[:,0:(shape[1]-shape[1]%windowsvert)]
    windowheight = int(shape[0]/windowsvert)
    windowwidth = int(shape[1]/windowslat)
    newshape = [windowsvert,windowslat]
    newshape.append(shape[0]//windowsvert)
    newshape.append(shape[1]//windowslat)
    images = np.zeros(newshape,dtype=np.uint8)
    for segv in range(0,windowsvert):
        for segl in range(0,windowslat):
            window = image[segv*windowheight:segv*windowheight+windowheight,segl*windowwidth:segl*windowwidth+windowwidth]
            images[segv,segl] = window
    return images


#binary bit setting operations used for LBP calculations
def ICV_get_bit(bin,index):
    return (bin >> index) & 1

def ICV_set_bit(bin,index):
    return bin | (1 << index)

def ICV_return_LBP(loc): #takes 3 by 3 array, and returns LBP value
    LBP = 0b00000000
    if loc[1,1]<loc[0,0]:
        LBP = ICV_set_bit(LBP,7)
    if loc[1,1]<loc[0,1]:
        LBP = ICV_set_bit(LBP,6)
    if loc[1,1]<loc[0,2]:
        LBP = ICV_set_bit(LBP,5)
    if loc[1,1]<loc[1,2]:
        LBP = ICV_set_bit(LBP,4)
    if loc[1,1]<loc[2,2]:
        LBP = ICV_set_bit(LBP,3)
    if loc[1,1]<loc[2,1]:
        LBP = ICV_set_bit(LBP,2)
    if loc[1,1]<loc[2,0]:
        LBP = ICV_set_bit(LBP,1)
    if loc[1,1]<loc[1,0]:
        LBP = ICV_set_bit(LBP,0)
    return LBP





def ICV_get_LBP_pattern(image):#finds the LBP value for each pixel and returns a new array of same shape as image wher pixel intensity is the LBP
    BP = np.zeros_like(image, dtype=np.uint8)
    height = BP.shape[0]
    width = BP.shape[1]
    for x in range(1,height-1):
        for y in range(1,width-1):
            loc = image[x-1:x+2,y-1:y+2]
            BP[x,y] = ICV_return_LBP(loc)
    BP = ICV_fill_border(BP)
    return BP

def ICV_image_histogram(image): #returns histogram with the frequency of each pixel intensity throughout an image
    hist = np.zeros((256,), dtype = int)
    for column in image:
        for pixel in column:
            hist[pixel] +=1
    return hist


def ICV_get_lbp(image):#takes an input image, splits into windows, calculates LBP of each window, finds histogram of LBP values for window, then concatenates them and returns the total histogram as an image descriptor
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vert = 2
    hori = 2
    windows = ICV_split_image(grey,vert,hori)
    total_hist = np.array([])
    for indv, row in enumerate(windows):
        for indh, window in enumerate(row):
            BP = ICV_get_LBP_pattern(window)
            mask = np.ones_like(BP)
            mask[0,:] = 0
            mask[-1,:] = 0
            mask[:,0] = 0
            mask[:,-1] = 0
            hist = cv2.calcHist([BP], [0], mask, [256], [0,255])
            if total_hist.shape == (0,):
                total_hist = hist
            else:
                total_hist = np.concatenate((total_hist, hist))
    return total_hist

def ICV_normalize_hist(hist): # takes a histogram (as a np.array) sums all the values, and then divides each histogram entry by this value, making it so that the total sums to 1
    tot = 0
    new_hist = np.zeros_like(hist)
    for a in hist:
        tot += a
    for index, a in enumerate(hist):
        new_hist[index] = a/tot
    return new_hist



def ICV_get_inter(image, ref, crit_val): #calculates intersection of values between two histograms
    ref_hist = ICV_get_lbp(ref)
    image_hist = ICV_get_lbp(image)
    ref_hist = ICV_normalize_hist(ref_hist)
    image_hist = ICV_normalize_hist(image_hist)
    intersect = cv2.compareHist(ref_hist, image_hist, cv2.HISTCMP_INTERSECT)
    if intersect>crit_val:
        return True, intersect
    return False, intersect



def run():
    car1 = cv2.imread('Dataset/DatasetA/car-1.jpg'); car2 = cv2.imread('Dataset/DatasetA/car-2.jpg'); car3 = cv2.imread('Dataset/DatasetA/car-3.jpg')
    
    face1 = cv2.imread('Dataset/DatasetA/face-1.jpg'); face2 = cv2.imread('Dataset/DatasetA/face-2.jpg'); face3 = cv2.imread('Dataset/DatasetA/face-3.jpg'); newface3 = cv2.imread('Dataset/DatasetA/newface-3.jpg')
    
    threshold = 0.85
    print(f'car 1 -- is car:{ICV_get_inter(car1, car1, threshold)}is face:{ICV_get_inter(car1, face1, threshold)}') 
    print(f'car 2 -- is car:{ICV_get_inter(car2, car1, threshold)}, is face:{ICV_get_inter(car2, face1, threshold)}')
    print(f'car 3 -- is car:{ICV_get_inter(car3, car1, threshold)}, is face:{ICV_get_inter(car3, face1, threshold)}')
    print(f'face 1 -- is car:{ICV_get_inter(face1, car1, threshold)}, is face:{ICV_get_inter(face1, face1, threshold)}')
    print(f'face 2 -- is car:{ICV_get_inter(face2, car1, threshold)}, is face:{ICV_get_inter(face2, face1, threshold)}')
    print(f'face 3 -- is car:{ICV_get_inter(face3, car1, threshold)}, is face:{ICV_get_inter(face3, face1, threshold)}')


if __name__ == '__main__':
    run()