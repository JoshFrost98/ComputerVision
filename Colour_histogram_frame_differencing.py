import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import statistics




def ICV_image_histogram(image): #calculates histogram for 2 dimensional array (1 channel of image)
    hist = np.zeros((256,), dtype = int)
    for column in image:
        for pixel in column:
            hist[pixel] +=1
    return hist



def ICV_import_image_rgb(name): #imports an image from file
    image = plt.imread(name)
    return image

def ICV_1_to_3_channels_rgb(image): # extracts rgb channels from image
    blue = image[:,:,0]
    green = image[:,:,1]
    red = image[:,:,2]
    return red, green, blue

def ICV_to_grey(r, g, b): # changes rgb image to greyscale
    image = 0.2989*r + 0.5870*g + 0.1140*b
    return image


def ICV_plot(hist, no): #plots values
    plt.figure(no)
    plt.plot(hist)

def ICV_hist_norm(hist, tot): # normalizes histogram by dividing by number of pixels
    hist_perc = np.zeros(256)
    for ind,a in enumerate(hist):
        hist_perc[ind] = a/tot
    return hist_perc

def ICV_intersect_hist(hist1, hist2): #finds intersection value between 2 histograms by taking the minimal value for each entry
    if hist1.shape == hist2.shape:
        intersect = np.zeros(256)
        tot = 0
        for index,bin in enumerate(hist1):
            a = min(hist1[index],hist2[index])
            intersect[index] = a
            tot += a
        return intersect, tot

def ICV_return_intersection(image1,image2, tot):  #returns normalized intersection for all three channels, RGB, between two given images
    c1 = ICV_1_to_3_channels_rgb(image1)
    c2 = ICV_1_to_3_channels_rgb(image2)
    intersections = []
    plots = np.zeros((3,256))
    for index in [0,1,2]:
        hist1,hist2 = ICV_hist_norm(ICV_image_histogram(c1[index]), tot),ICV_hist_norm(ICV_image_histogram(c2[index]), tot)
        plot, intersect = ICV_intersect_hist(hist1,hist2)
        intersections.append(intersect)
        plots[index] = np.array(plot)
    tot_intersection = statistics.mean(intersections)
    return plots, tot_intersection

def ICV_import_video(path):  #imports a video
    cap = cv2.VideoCapture(path)
    inters = []
    frameno = 0
    while(True) :
        ret,frame = cap.read() #reads frame by frame
        

        if ret == True:
            if frameno == 0:#initializes values for first frame
                newframe = np.array(frame)
                tot = newframe.shape[0] * newframe.shape[1] #total number of pixels
            if frameno > 0:   #for subsequent frames, calculates intersection between current frame and previous frame
                vals,intersect_value, = ICV_return_intersection(prev,frame, tot)
                inters.append(intersect_value)
            prev = frame
            frameno += 1
            print(f'frame {frameno}')
        else:
            break
    return inters, tot


def run():
    inters, tot = ICV_import_video('Dataset/DatasetB.avi')
    #finds normalized intersection values for each frame of DatasetB
    
    ICV_plot(inters,1) #plots intersection value for each subsequent frame
    plt.xlabel('Frame number')
    plt.ylabel('intersection value')
    plt.savefig('intersection_vals_non.png')
    plt.show()




if __name__ == '__main__':
    run()