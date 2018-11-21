import cv2
import numpy as np
import glob
import os
import random


path = "/media/antor/Files/main_projects/other_org/" + "*.jpg"
dir_list = glob.glob(path)

base_lst = [os.path.basename(x) for x in dir_list]
name_lst = [os.path.splitext(x)[0] for x in base_lst]
int_lst = list(map(int,name_lst))
int_lst.sort()
str_lst = list(map(str,int_lst))

ran_list = []

for i in range(int(len(str_lst)/6)):
    obj_lst = list(str_lst[6*i:6*(i+1)])
    ran_list.append(obj_lst)

###seeded with 12 for psudorandom generator
seed_ran = 12
###randomize list
random.seed(seed_ran)
random.shuffle(ran_list)
#print(ran_list)
#print(len(ran_list))
coun = 0
cou = 0
an = 0

for i in range(144):
    if i==72:
        coun = 0
        cou = 0
        an = 0

    for j in range(6):

        img = cv2.imread("/media/antor/Files/main_projects/other_org/" + ran_list[i][j] + ".jpg", cv2.IMREAD_GRAYSCALE)
        #print(i)
        for f in range(3):
            coun = cou+(6*f)
            crop_img = img[0:256,256*f:256*(f+1)]

            if i<72:
                cv2.imwrite("/media/antor/Files/main_projects/dev_set/"+str(coun)+".jpg", crop_img)
            else:

                cv2.imwrite("/media/antor/Files/main_projects/test_set/"+str(coun)+".jpg", crop_img)
            print(coun)
            #print(coun)
        cou = cou+1

        an = an+1
        if an==6:
            cou=cou+12
            an=0
