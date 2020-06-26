import cv2
import os
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import numpy as np

import io
import os

fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),"crop.png")

image = cv2.imread(fn)
image = image[34:85, :]
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([153, 31, 139])
upper_hsv = np.array([173, 46, 255])
mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
cv2.imshow('mask Image',mask)
print(pytesseract.image_to_string(mask))
#自定义卷积核
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
kernel_sharpen_2 = np.array([
        [1,1,1],
        [1,-7,1],
        [1,1,1]])
kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1], 
        [-1,-1,-1,-1,-1]])/8.0
#卷积
output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)

#显示锐化效果
cv2.imshow('Original Image',image)
cv2.imshow('sharpen_1 Image',output_1)
cv2.imshow('sharpen_2 Image',output_2)
cv2.imshow('sharpen_3 Image',output_3)
output_3 = cv2.cvtColor(output_3, cv2.COLOR_BGR2GRAY)

ret,thresh1 = cv2.threshold(output_3,190,255,cv2.THRESH_BINARY)
cv2.imshow("Binary", thresh1)
print("--------Binary---------------")
print(pytesseract.image_to_string(thresh1))

#高斯滤波
print("Gaussian")
g1=cv2.GaussianBlur(output_3,(3,3),0)
print("----------------------------")
print(pytesseract.image_to_string(g1))
g2=cv2.GaussianBlur(output_3,(5,5),0)
print("----------------------------")
print(pytesseract.image_to_string(g2))
g3=cv2.GaussianBlur(output_3,(7,7),0)
print("----------------------------")
print(pytesseract.image_to_string(g3))
blurred = np.hstack([g1,
                     g2,
                     g3
                     ])
cv2.imshow("Gaussian",blurred)
print("Gaussian Merge")
print(pytesseract.image_to_string(blurred))
#中值滤波

m1 = cv2.medianBlur(output_3,3)
print("Median")
print("----------------------------")
print(pytesseract.image_to_string(m1))
m2 = cv2.medianBlur(output_3,5)
print("----------------------------")
print(pytesseract.image_to_string(m2))
m3 = cv2.medianBlur(output_3,7)
print("----------------------------")
print(pytesseract.image_to_string(m3))
blurred = np.hstack([m1,
                     m2,
                     m3
                     ])
cv2.imshow("Median",blurred)
print("Median merger")
print(pytesseract.image_to_string(blurred))

#双边滤波
aa = cv2.bilateralFilter(output_3,5,21,21)
bb = cv2.bilateralFilter(output_3,7, 31, 31)
cc = cv2.bilateralFilter(output_3,9, 41, 41)
blurred = np.hstack([aa,
                     bb,
                     cc
                     ])
cv2.imshow("Bilateral",blurred)
print("Bilateral")
print(pytesseract.image_to_string(blurred))

#停顿
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
