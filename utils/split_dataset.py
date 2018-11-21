import cv2
import numpy as np
import glob
import os

path = "/media/antor/Files/main_projects/final_data/" + "*.jpg"
dir_list = glob.glob(path)

###make a list of basenames in str from the total path
base_lst = [os.path.basename(x) for x in dir_list]
#rint(base_lst)
###make a list of filenames as str
name_lst = [os.path.splitext(x)[0] for x in base_lst]
#print(name_lst)

###make int list of names
int_lst = list(map(int,name_lst))
###sort the int list
int_lst.sort()
###convert the list of ints' back to list of strs'
str_lst = list(map(str,int_lst))

print(str_lst)
coun = 0
cou = 0
an = 0

for i in range(len(str_lst)):
    img = cv2.imread("/media/antor/Files/main_projects/final_data/" + str_lst[i] + ".jpg", cv2.IMREAD_GRAYSCALE)

    print(i)

    if an in range(24, 30 ,1):
        cv2.imwrite("/media/antor/Files/main_projects/other_org/" + str(i) + ".jpg", img)
    else:
        for f in range(3):
            coun = cou+(30*f)
            crop_img = img[0:256,256*f:256*(f+1)]
            cv2.imwrite("/media/antor/Files/main_projects/train_org/"+str(coun)+".jpg", crop_img)
            print(coun)


            #print(coun)
        cou = cou+1

    an = an+1
    if an==30:
        cou=cou+60
        an=0
