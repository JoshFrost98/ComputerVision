import cv2
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import math

# -------these functions all simply return a 3 by 3 transformation matrix for-------
# rotation matrix for angle phi
def ICV_rot_t_mat(phi):
    rot_mat = np.array([
        [math.cos(phi), math.sin(phi), 0],
        [-math.sin(phi), math.cos(phi), 0],
        [0, 0, 1]
        ])
    return rot_mat

# inverse rotation matrix for angle phi
def ICV_rot_t_mat_i(phi):
    rot_mat = np.linalg.inv(np.array([
        [math.cos(phi), math.sin(phi), 0],
        [-math.sin(phi), math.cos(phi), 0],
        [0, 0, 1]
        ]))
    
    return rot_mat

# skew matrix for angle theta (theta is the angle the made with the y axis)
def ICV_skew_t_mat(theta):
    rot_mat = np.array([
        [1,0, 0],
        [-math.tan(theta), 1, 0],
        [0, 0, 1]
        ])
    return rot_mat

# inverse skew matrix for angle theta (theta is the angle the made with the y axis)
def ICV_skew_t_mat_i(theta):
    rot_mat = np.linalg.inv(np.array([
        [1, 0, 0],
        [-math.tan(theta), 1, 0],
        [0, 0, 1]
        ]))
    return rot_mat

# matrix to shift vector by x and y
def ICV_shift_mat(x, y):
    rot_mat = np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
        ])
    return rot_mat

# these functions are for assigning a grey value to an interpolated location
# this function uses bilinear interpolation, taking as input the input coordinates and the image from which the value must be found
# this function returns just the grey value for these input oordinates
def ICV_bilinear_interpolation(pixel_coords, original):
    x, y = pixel_coords[0], pixel_coords[1]
    x1,y1= math.floor(x), math.floor(y)
    x2,y2 = x1+1, y1+1
    result = (original[x1, y1] * (x2 - x) * (y2 - y) +
            original[x2, y1] * (x- x1) * (y2 - y) +
            original[x1, y2] * (x2 - x) * (y - y1) +
            original[x2, y2] * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1))
    return int(round(result))

# this function copies the grey value from the closest pixel in the original image
def ICV_nearest_neighbors(pixel_coords, original):
    x, y = pixel_coords[0], pixel_coords[1]
    return original[round(x), round(y)]


# this function uses the four corners of the original image,
# returning the shape of the final transformed image.
def ICV_calc_shift(phi, theta, image):
    width, height = int(round(image.shape[0])), int(round(image.shape[1]))
    corners = [
        [0,0, 1],
        [0,height, 1],
        [width,0, 1],
        [width, height, 1],
    ]
    n_corners = []
    s = ICV_skew_t_mat(theta)
    r = ICV_rot_t_mat(phi)
    sh = ICV_shift_mat(-width/2, -height/2)
    n_corners = np.dot(np.dot(np.dot(s,r), sh), np.transpose(corners))
    n_corners = np.transpose(n_corners)
    x_tot = int(round(max(n_corners[:,0]) - min(n_corners[:,0])))
    y_tot = int(round(max(n_corners[:,1]) - min(n_corners[:,1])))
    return x_tot, y_tot

# this function takes an image and returns an array containing coordinates for each pixel
def ICV_coords_to_array(new_image):
    coords = []
    for x in range(0,new_image.shape[0]):
        for y in range(0,new_image.shape[1]):
            coords.append([x,y,1])
    return coords

# this function iterates through each pixel in the new image, for each pixel in the function is given a mapped coordinate on the original image
# this function then calls the bilinear interpolation or the nearest neighbors function to assign a value for each pixel
def ICV_fill_pixels(coords, orig_coords, image, x_dim, y_dim, x_dim1, y_dim1):
    new_image = np.empty((x_dim1+1,y_dim1+1))
    for pixel in range(0,len(coords)):
        x, y, x1, y1 = coords[pixel][0], coords[pixel][1], orig_coords[pixel][0], orig_coords[pixel][1]
        if 0 <= x1 < x_dim-1:
            if 0 <= y1 < y_dim-1:
                new_image[x,y] = ICV_bilinear_interpolation([x1, y1], image)
    return new_image


# this function applies a given rotation and then skew to a given image, it returns a new transformed image
# input phi or theta = 0 to just perform rotation or skew
def ICV_apply_t(image, phi, theta):
    x_dim1, y_dim1 = ICV_calc_shift(phi, theta, image)#shape of new image
    x_dim, y_dim = image.shape[0], image.shape[1]#shape of original image
    new_image = np.empty((x_dim1+1,y_dim1+1))#assigning space for new image
    sh1 = ICV_shift_mat(x_dim/2, y_dim/2)#matrix to shift original image to centre
    ri = ICV_rot_t_mat_i(phi)#inverse rotation matrix
    si = ICV_skew_t_mat_i(theta)#inverse skew matrix
    sh = ICV_shift_mat(-x_dim1/2, -y_dim1/2)#matrix to shift transformed image(at centre) back out to correct coordinates
    coords = ICV_coords_to_array(new_image)#array of coordinates in new image
    orig_coords = np.transpose(np.dot(np.dot(np.dot(np.dot(sh1, ri) ,si) , sh), np.transpose(coords)))#transformed coordinates of pixels from new image onto original image
    new_image = ICV_fill_pixels(coords, orig_coords, image, x_dim, y_dim, x_dim1, y_dim1)
    return(new_image)

# imports an image from file and returns red, blue, and green channels
def ICV_import_image_rgb(name):
    image = cv2.imread(name, cv2.IMREAD_COLOR)
    blue = image[:,:,0]
    green = image[:,:,1]
    red = image[:,:,2]
    return red, green, blue
    

# runs the above functions for each channel in rgb, combines the three rotated channels and displays the new rotated image
def run():
    r, g, b = ICV_import_image_rgb('name.png')
    
    phi = 20*math.pi/180 #rotation angle
    theta = 50*math.pi/180 #skew angle
    r1 = ICV_apply_t(r, phi, theta)
    g1 = ICV_apply_t(g, phi, theta)
    b1 = ICV_apply_t(b, phi, theta)
    new_image = np.empty((b1.shape[0],b1.shape[1],3))
    new_image[:,:,0] = b1
    new_image[:,:,1] = g1
    new_image[:,:,2] = r1
    new_image = new_image.astype(np.uint8)
    cv2.imwrite('skewrot.png', new_image)
    cv2.imshow('hello', new_image)
    cv2.waitKey(0)
    print('running')

run()