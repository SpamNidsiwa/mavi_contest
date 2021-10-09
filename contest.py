import numpy as np
import cv2
import glob
import csv
from numpy.core.arrayprint import printoptions
from pandas.core import frame

with open('result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name","Meat","Veggie","Noodle"])

path = glob.glob('ranking_round\images\*.jpg')

for n in path:
    img = cv2.imread(n)
    name = n[21::]
    print(name)
    print(len(img)*len(cv2.transpose(img)))
    nPixel = len(img)*len(cv2.transpose(img))

    hsv_frame = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernel2 = np.ones((2,2),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)
    kernel5 = np.ones((5,5),np.uint8)
    kernel10 = np.ones((10,10),np.uint8)

    ##########################################################################

    low_meat = np.array([1,114,10])
    high_meat = np.array([19,160,90])
    mask_meat = cv2.inRange(hsv_frame,low_meat,high_meat)
    ero_meat = cv2.erode(mask_meat,kernel2,iterations=1)
    di_meat = cv2.dilate(ero_meat,kernel2,iterations=1)
    di_meat = cv2.dilate(ero_meat,kernel10,iterations=1)
    ero_meat = cv2.erode(di_meat,kernel10,iterations=1)

    low_veg = np.array([27,90,30])
    high_veg = np.array([80,255,200])
    mask_veg = cv2.inRange(hsv_frame,low_veg,high_veg)
    ero_veg = cv2.erode(mask_veg,kernel2,iterations=1)
    di_veg = cv2.dilate(ero_veg,kernel3,iterations=1)
    ero_veg = cv2.erode(di_veg,kernel3,iterations=1)

    low_cab = np.array([5,105,90])
    high_cab = np.array([26,255,255])
    mask_cab = cv2.inRange(hsv_frame,low_cab,high_cab)
    ero_cab = cv2.erode(mask_cab,kernel2,iterations=1)
    di_cab = cv2.dilate(ero_cab,kernel10,iterations=1)
    ero_cab = cv2.erode(di_cab,kernel10,iterations=1)

############################################################################

    meat = ("{:.2f}".format(cv2.countNonZero(ero_meat)/nPixel*275))
    veg = ("{:.2f}".format(cv2.countNonZero(ero_veg)/nPixel*275))
    nood = ("{:.2f}".format(cv2.countNonZero(ero_cab)/nPixel*275))


    with open('result.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name,meat,veg,nood])

    cv2.imshow("Image",img)
    cv2.imshow("mask",mask_cab)
    cv2.imshow("e",ero_cab)

cv2.waitKey(0)
cv2.destroyAllWindows()