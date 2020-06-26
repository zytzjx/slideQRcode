import cv2,time
import numpy as np
import os
import pytesseract

#global variable to keep track of 
show = False

def onTrackbarActivity(x):
    global show
    show = True
    pass

######
#  crop 191-255, binary. it is right IMEI.
if __name__ == '__main__' :
    fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),"crop.png")

    image = cv2.imread('imageOCR/1729.png')
    #image = image[34:85, :]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([153, 31, 139])
    upper_hsv = np.array([173, 46, 255])
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    cv2.imshow('mask Image',mask)
    #print(pytesseract.image_to_string(mask))
    
    kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1], 
        [-1,-1,-1,-1,-1]])/8.0
    #卷积qa
    output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
    #output_3 = cv2.morphologyEx(output_3, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    original = cv2.cvtColor(output_3, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", original)
    '''
    th2 = cv2.adaptiveThreshold(original,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2) #换行符号 
    cv2.imshow("Mean_c", th2)
    print("-----Mean_c-------")
    print(pytesseract.image_to_string(th2))
    th3 = cv2.adaptiveThreshold(original,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2) #换行符号 
    cv2.imshow("GAUSSIAN_c", th3)
    print("------GAUSSIAN_c------")
    print(pytesseract.image_to_string(th3))
    '''
    cv2.namedWindow('binary',cv2.WINDOW_AUTOSIZE)   
    cv2.createTrackbar('Min','binary',0,255,onTrackbarActivity)
    cv2.createTrackbar('Max','binary',0,255,onTrackbarActivity)

    cv2.imshow('binary',original)
    #while cv2.waitKey(0) & 0xFF == 27:
    #    break
    cv2.setTrackbarPos('Min', 'binary', 189)
    cv2.setTrackbarPos('Max', 'binary', 255)
    index = 0
    cong = r'--oem 3 --psm 6 outputbase digits'
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if show: # If there is any event on the trackbar
            show = False
            
            Min = cv2.getTrackbarPos('Min','binary')
            Max = cv2.getTrackbarPos('Max','binary')
            ret,thresh1 = cv2.threshold(original,Min,Max,cv2.THRESH_BINARY)
            #thresh1 = cv2.medianBlur(thresh1, 3)
            cv2.imshow("Binary", thresh1)
            print("------------")
            print(pytesseract.image_to_string(thresh1, config=cong))
            cv2.imwrite("images/%d.png" % index, thresh1)
            index += 1

    cv2.destroyAllWindows()