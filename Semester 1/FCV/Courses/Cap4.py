# FCV C4 examples
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


# Prewitt
if(ex==1):
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image    
    cv2.imshow("Img",imgGray)
    
    kernel = np.array([[1,   1,  1],
                       [0,   0,  0],
                       [-1, -1, -1]])  
    imgPrewH = cv2.filter2D(imgGray,-1,kernel)
    cv2.imshow("Prewitt H",imgPrewH)
    cv2.imshow("Prewitt H negate",cv2.bitwise_not(imgPrewH))  

    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
    imgPrewV = cv2.filter2D(imgGray,-1,kernel)
    cv2.imshow("Prewitt V",imgPrewV)
    cv2.imshow("Prewitt V negate",cv2.bitwise_not(imgPrewV))

    imgSum = imgPrewH + imgPrewV
    cv2.imshow("Prewitt sum negate",cv2.bitwise_not(imgSum))
    cv2.waitKey()
    cv2.destroyAllWindows()


# Sobel
if(ex==2):
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image    
    cv2.imshow("Img",imgGray)
    
    kernel = np.array([[1,   2,  1],
                       [0,   0,  0],
                       [-1, -2, -1]])  # /8.0
    imgSobelH = cv2.filter2D(imgGray,-1,kernel)
    cv2.imshow("Sobel H",imgSobelH)
    cv2.imshow("Sobel H negate",cv2.bitwise_not(imgSobelH))  

    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    imgSobelV = cv2.filter2D(imgGray,-1,kernel)    
    cv2.imshow("Sobel V",imgSobelV)   
    cv2.imshow("Sobel V negate",cv2.bitwise_not(imgSobelV))

    imgSum = imgSobelH + imgSobelV
    cv2.imshow("Sobel sum negate",cv2.bitwise_not(imgSum))

    cv2.waitKey()
    cv2.destroyAllWindows()


# Roberts
if(ex==3):
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image    
    cv2.imshow("Img",imgGray)
    
    kernel = np.array([[0, 0,  0],
                       [0, -1, 0],
                       [0, 0,  1]])  
    imgRobH = cv2.filter2D(imgGray,-1,kernel)
    cv2.imshow("Roberts H",imgRobH)
    cv2.imshow("Roberts H negate",cv2.bitwise_not(imgRobH))  

    kernel = np.array([[0, 0,  0],
                       [0, 0, -1],
                       [0, 1,  0]])
    imgRobV = cv2.filter2D(imgGray,-1,kernel)
    cv2.imshow("Roberts V",imgRobV)
    cv2.imshow("Roberts V negate",cv2.bitwise_not(imgRobV))

    #ret, imgBW = cv2.threshold(imgRobV, 10, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Roberts V threshold negate",cv2.bitwise_not(imgBW))

    imgSum = imgRobH + imgRobV
    cv2.imshow("Roberts sum negate",cv2.bitwise_not(imgSum))

    cv2.waitKey()
    cv2.destroyAllWindows()

# Canny
if(ex==4):
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image    
    cv2.imshow("Img",imgGray)
    
    #imgCanny = cv2.Canny(imgGray, 50, 120, 3)
    #cv2.imshow("Canny",imgCanny)
    #cv2.imshow("Canny negate",cv2.bitwise_not(imgCanny))

    imgGray = cv2.GaussianBlur(imgGray,(7,7), 0) 
    cv2.imshow("Img Blurred",imgGray)
    imgCanny = cv2.Canny(imgGray, 50, 50, 3)
    cv2.imshow("Canny blurred negate",cv2.bitwise_not(imgCanny))  

    cv2.waitKey()
    cv2.destroyAllWindows()

# Hough
if(ex==5):
    img = cv2.imread(img_file)
    imgGray = cv2.imread(img_file, 0)  # read a grayscale image    
    cv2.imshow("Img",imgGray)
    
    edges = cv2.Canny(imgGray, 10, 250, apertureSize = 3)
    cv2.imshow("Edges", edges)
    minLineLength = 80
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 2, np.pi/720, 100, minLineLength, maxLineGap)
    for lin in lines:
        for x1,y1,x2,y2 in lin:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("Hough lines", img)  

    cv2.waitKey()
    cv2.destroyAllWindows()

exit()