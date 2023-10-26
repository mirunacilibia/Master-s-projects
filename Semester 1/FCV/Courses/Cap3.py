# FCV C2 examples
# Dan Pescaru, 2021
import cv2
import numpy as np
import tkinter.filedialog
from matplotlib import pyplot as plt

ex = 1
th = 140

root = tkinter.Tk()
img_file = tkinter.filedialog.askopenfilename(initialdir=".", title="Select image file", filetypes=(("Image files", "*.jpg;*.png"), ("all files", "*.*")))
root.destroy()

# erode / dilate on BW image
if(ex==1):
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image
    ret, imgBW = cv2.threshold(imgGray, th, 255, cv2.THRESH_BINARY)    
    imgBW = cv2.bitwise_not(imgBW)
    cv2.imshow("Img",imgBW)

    kernel = np.ones((5,5),np.uint8)
    imgErode = cv2.erode(imgBW,kernel,iterations = 1)
    cv2.imshow("Erode",imgErode)
    imgDilate = cv2.dilate(imgBW,kernel,iterations = 1)
    cv2.imshow("Dilate",imgDilate)
    imgOpen = cv2.morphologyEx(imgBW, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Open",imgOpen)
    imgClose = cv2.morphologyEx(imgBW, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Close",imgClose)
    imgGradient = cv2.morphologyEx(imgBW, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Mgradient",imgGradient)
    cv2.waitKey()
    cv2.destroyAllWindows()

#on negated BW image
if(ex==2):
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image
    ret, imgBW = cv2.threshold(imgGray, th, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Img inv",cv2.bitwise_not(imgBW))

    kernel = np.ones((6,6),np.uint8)
    #kernel = np.ones((1,6),np.uint8)
    #kernel = np.ones((6,1),np.uint8)
    imgErode = cv2.erode(imgBW,kernel,iterations = 1)
    cv2.imshow("Erode",cv2.bitwise_not(imgErode))
    imgDilate = cv2.dilate(imgBW,kernel,iterations = 1)
    cv2.imshow("Dilate",cv2.bitwise_not(imgDilate))
    imgOpen = cv2.morphologyEx(imgBW, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Open",cv2.bitwise_not(imgOpen))
    imgClose = cv2.morphologyEx(imgBW, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Close",cv2.bitwise_not(imgClose))
    imgGradient = cv2.morphologyEx(imgBW, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Mgradient",cv2.bitwise_not(imgGradient))
    cv2.waitKey()
    cv2.destroyAllWindows()


# on Grayscale
if(ex==3):
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image
    cv2.imshow("Img",imgGray)
    kernel = np.ones((5,5),np.uint8)
    imgErode = cv2.erode(imgGray,kernel,iterations = 1)
    cv2.imshow("Erode",imgErode)
    imgDilate = cv2.dilate(imgGray,kernel,iterations = 1)
    cv2.imshow("Dilate",imgDilate)
    imgOpen = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Open",imgOpen)
    imgClose = cv2.morphologyEx(imgGray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Close",imgClose)
    imgGradient = cv2.morphologyEx(imgGray, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Mgradient",imgGradient)
    cv2.waitKey()
    cv2.destroyAllWindows()
