import numpy as np
import math


def ICV_fill_border(image): #fills border of image using neighboring values
    image = np.array(image)
    x =len(image[:,0])
    y =len(image[0])
    image[0,:] = image[1,:]
    image[x-1,:] = image[x-2,:]
    image[:,0] = image[:,1]
    image[:,y-1] = image[:,y-2]
    return image


def  ICV_apply_kernel(image, convolution_matrix): #applies kernel to each pixel in image, requires kernel to be 3 by 3
    new_image = np.zeros(np.shape(image))
    if np.shape(convolution_matrix) == (3,3):
        min = 0
        max = 0
        for x in range(1,image.shape[0]-1): 
            for y in range(1,image.shape[1]-1): #iterates through each pixel
                new_image[x][y] =(
                    image[x-1][y-1] * convolution_matrix[0][0] +
                    image[x-1][y] * convolution_matrix[0][1] +
                    image[x-1][y+1] * convolution_matrix[0][2] +
                    image[x][y-1] * convolution_matrix[1][0] +
                    image[x][y] * convolution_matrix[1][1] +
                    image[x][y+1] * convolution_matrix[1][2] +
                    image[x+1][y-1] * convolution_matrix[2][0] +
                    image[x+1][y] * convolution_matrix[2][1] +
                    image[x+1][y+1] * convolution_matrix[2][2] 
                )
                if new_image[x][y]>255:#currently clamps all values between 0 and 255 eg. 34 stays the same, but a value of 300 would be clamped to 255
                    new_image[x][y] = 255
                if new_image[x][y]<0:
                    new_image[x][y] = 0
        return ICV_fill_border(new_image)
    else:
        return('convolution matrix not 3 by 3')