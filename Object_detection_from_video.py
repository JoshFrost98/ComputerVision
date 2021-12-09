import numpy as np
import matplotlib.pyplot as plt
import math
import cv2



def ICV_get_pixel_differences(frame, prev): # returns an array where each value is the difference between frame and prev
    out = np.zeros([frame.shape[0], frame.shape[1]])
    for x, column in enumerate(frame):
        for y, pixel in enumerate(column):
            out[x,y] = int(frame[x,y]) - int(prev[x,y])
    return out

def ICV_clamp_differences(difference, threshold): # clamps differences based on a threshold value. output binary mask
    out = np.full([difference.shape[0], difference.shape[1]], 255)
    for x, column in enumerate(difference):
        for y, pixel in enumerate(column):
            if pixel>threshold:
                out[x,y] = 0
            if pixel<-threshold:
                out[x,y] = 0
    return out

def ICV_fill_pixels(frame, x, y, search_val): #self created flood fill algorithm, it does work but is slow so cv2 algorithm used instead.
    frame[x,y] = 0
    maxx = frame.shape[0]-1
    maxy = frame.shape[1]-1
    if x>0 and x < maxx and y>0 and y<maxy:
        if frame[x+1, y] == search_val:
            frame = ICV_fill_pixels(frame,  x+1, y, search_val)
        if frame[x, y+1] == search_val:
            frame = ICV_fill_pixels(frame, x+1, y, search_val)
        if frame[x-1, y] == search_val:
            frame = ICV_fill_pixels(frame, x+1, y, search_val)
        if frame[x, y-1] == search_val:
            frame = ICV_fill_pixels(frame, x+1, y, search_val)
    return frame


def ICV_object_detection(orig): #fills regions in binary mask with count, then returns total number of objects in frame
    frame = np.copy(orig)
    count = 0
    for x, column in enumerate(frame):
        for y, pixel in enumerate(column):
            if pixel == 255:
                cv2.floodFill(frame, None, (y,x), count)
                count+=1
    print(count)
    return count

def invert(a): #inverts values eg. 255 goes to 0 and vice versa
    for x, b in enumerate(a):
        for y, c in enumerate(b):
            if c>125:
                a[x,y] = 0
            else:
                a[x,y]=255
    return a

def Clamp(a): 
    for x, b in enumerate(a):
        for y, c in enumerate(b):
            if c>125:
                a[x,y] = 255
            else:
                a[x,y]=0
    return a

def ICV_dilate(image, kernel): #dilate function with kernel size 5 by 5
    new_image = np.zeros_like(image)
    for x, column in enumerate(image[2:-2]):
        for y, val in enumerate(column[2:-2]):
            section = image[x:x+5, y:y+5]
            tot = np.multiply(section, kernel)
            if tot.any()>0:
                new_image[x+2,y+2] = 255
    return new_image

def ICV_dilates(image, kernel): #dilate function with kernel size 3 by 3
    new_image = np.zeros_like(image)
    for x, column in enumerate(image[1:-1]):
        for y, val in enumerate(column[1:-1]):
            section = image[x:x+3, y:y+3]
            tot = np.multiply(section, kernel)
            if tot.any()>0:
                new_image[x+1,y+1] = 255
    return new_image




def ICV_frame_differencing_beginning(path):
    threshold = 20
    cap = cv2.VideoCapture(path)
    frameno = 0
    background = np.zeros([288,352])
    prev_classification = np.array([[255]])
    no_objects = 0
    background = cv2.cvtColor(cv2.imread('background.png'), cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(background.astype(np.uint8),(5,5),0)
    object_count = []
    while(True) :
        ret,frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #format and filter frame before processing
            frame = cv2.GaussianBlur(frame.astype(np.uint8),(5,5),0) 
            # frame = cv2.medianBlur(frame.astype(np.uint8),5)#UNHASH THESE TWO LINES FOR BETTER OBJECT COUNTING, but its slower
            
            if frameno > 1:
                difference = ICV_get_pixel_differences(frame, prev)   #call difference function
                classification = invert(ICV_clamp_differences(difference, threshold)).astype(np.uint8) #use differences to create binary mask classification
                classification = Clamp(cv2.medianBlur(classification.astype(np.uint8),5))#median filtering
                kernelb = np.ones((7, 7), np.uint8)
                kernelm = np.ones((5, 5), np.uint8)#kernels for dilation
                kernels = np.ones((3, 3), np.uint8)
                classification = ICV_dilate(classification, kernelm) #dilation applied twice to classification mask
                # classification = ICV_dilate(classification, kernelm) #UNHASH FOR BETTER OBJECT COUNTING
                
                cv2.imshow('Frame', classification.astype(np.uint8))
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
                no_objects = ICV_object_detection(classification) #use mask to count objects
                object_count.append(no_objects) #create list of object counts
            else:
                orig = frame #initializes original frame
            prev = frame #updates previous frame
            frameno += 1
        else:
            plt.bar(range(0,len(object_count)), object_count)   #plots object counts
            plt.xlabel('frame')
            plt.ylabel('Object count')
            plt.savefig('object_count.png')
            plt.show()
            break

#functions very similarly to previous function, but calculates and outputs a background.
def ICV_make_background(path):
    threshold = 5
    cap = cv2.VideoCapture(path)
    frameno = 0
    background = np.zeros([288,352, 3])
    prev_classification = np.array([[255]])

    while(True) :
        ret,frame = cap.read()
        if ret == True:
            frame_orig = np.copy(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame.astype(np.uint8),(5,5),0)
            frame = cv2.medianBlur(frame.astype(np.uint8),5)
            if frameno > 1:
                difference = ICV_get_pixel_differences(frame, prev)
                classification = ICV_clamp_differences(difference, threshold)
                cv2.imshow('Frame', frame_orig)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
                mask = np.transpose([np.transpose(classification), np.transpose(classification), np.transpose(classification)]) #reformats binary mask into 3 channels for use over RGB image
                background = np.add(background, np.multiply(mask, frame_orig/255))
            else:
                orig = frame
            prev = frame
            frameno += 1
        else:
            background = background/frameno #Normalizing background intensities to correct scale
            cv2.imshow('backg',background)
            cv2.imwrite('background.png', background.astype(np.uint8))
            break

def run():
    # ICV_make_background('Dataset/DatasetC.mpg')
    ICV_frame_differencing_beginning('Dataset/DatasetC.mpg') #currently just doing object counting, not making a background


if __name__ == '__main__':
    run()