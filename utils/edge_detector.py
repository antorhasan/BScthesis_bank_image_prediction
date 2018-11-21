import cv2
import numpy as np
#import matplotlib.pyplot as plt
import glob
import os

path = "/media/antor/Files/main_projects/dev_set/*.jpg"
dir_list = glob.glob(path)
base_lst = [os.path.basename(x) for x in dir_list]
name_lst = [os.path.splitext(x)[0] for x in base_lst]
int_lst = list(map(int,name_lst))
int_lst.sort()
str_lst = list(map(str,int_lst))

str_lst = str_lst[0:6]

#print(len(str_lst))

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

counter = 1
for i in range(len(str_lst)):
    #img = cv2.imread("/media/antor/Files/main_projects/dev_set/" + str_lst[i] +".jpg",cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("/media/antor/Files/main_projects/dev_set/" + str_lst[i] +".jpg")

    #denoise = cv2.fastNlMeansDenoising(img, h=24, templateWindowSize=7, searchWindowSize=21)
    blur = cv2.pyrMeanShiftFiltering(img,75,1)
    edges = auto_canny(blur)
    _, contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    cv2.drawContours(img, contours, -1, (0,255,0), 1)
    #denoise = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    cv2.namedWindow('denoise',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('denoise',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite("F:/D/Papers/River/"+str(counter)+".jpg",edges)
    counter+=1
