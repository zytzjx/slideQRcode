# -*- coding: utf-8 -*-
# feimengjuan
import numpy as np
import cv2
import math
import pytesseract

 
def image_resize(image, width = None, height = None, ratio=0.0, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None and math.isclose(ratio, 0.0):
        return image

    if math.isclose(ratio, 0.0):
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
    else:
        dim = (int(w * ratio), int(h * ratio))
        print(dim)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



# 1、整个条形码的算法流程如下：
# 2、计算x方向和y方向上的Scharr梯度幅值表示
# 3、将x-gradient减去y-gradient来显示条形码区域
# 4、模糊并二值化图像
# 5、对二值化图像应用闭运算内核
# 6、进行系列的腐蚀、膨胀
# 7、找到图像中的最大轮廓，大概便是条形码
# 注：该方法做了关于图像梯度表示的假设，因此只对水平条形码有效。

def detect_bar(gray):
    # 读入图片并灰度化
    cv2.imshow("gray",image_resize(gray, ratio=1.0))
    # 计算图像x方向和y方向的梯度
    gradX = cv2.Sobel(gray,ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray,ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    # 利用x-gradient减去y-gradient，通过这一步减法操作，得到包含高水平梯度和低竖直梯度的图形区域
    gradient = cv2.subtract(gradX,gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # 利用去噪仅关注条形码区域，使用9*9的内核对梯度图进行平均模糊，
    # 有助于平滑梯度表征的图形中的高频噪声，然后进行二值化
    blurred = cv2.blur(gradient,(9,9))
    cv2.imshow("blurred",image_resize(blurred, ratio=1.0))
    (_,thresh) = cv2.threshold(blurred,128,255,cv2.THRESH_BINARY)
    cv2.imshow("thresh",image_resize(thresh, ratio=1.0))

    # 对二值化图进行形态学操作，消除条形码竖杠之间的缝隙
    # 使用cv2.getStructuringElement构造一个长方形内核。这个内核的宽度大于长度，
    # 因此我们可以消除条形码中垂直条之间的缝隙。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
    closed = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("closed",image_resize(closed, ratio=1.0))
   
    # 然后图像中还存在一些小斑点，于是用腐蚀和膨胀来消除旁边的小斑点
    #  腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    #  而膨胀操作将使剩余的白色像素扩张并重新增长回去。
    closed = cv2.dilate(closed,None,iterations = 4)
    closed = cv2.erode(closed,None,iterations = 4)
    closed = cv2.dilate(closed,None,iterations = 4)
    closed = cv2.erode(closed,None,iterations = 4)
    cv2.imshow("closed1",image_resize(closed, ratio=1.0))
    #cv2.waitKey(0)
    # 最后找图像中国条形码的轮廓
    (cnts,_) = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # 通过对轮廓面积进行排序，找到面积最大的轮廓即为最外层轮廓
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # 计算最大轮廓的包围box
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # 将box画在原始图像中显示出来，这样便成功检测到了条形码
    cv2.drawContours(image,[box],-1,(0,255,0),3)

    contours, hier = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = 0, 0, 0, 0
    if len(contours)>0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,255),1)

    return image

def featuredetect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 15, 5, 5)
    gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray,2,3,0.4)

    # #result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)

    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.1*dst.max()]=[0,0,255]


    # height, width = gray.shape
    # dst = np.zeros((height,width,1),np.uint8)
    # for i in range(height):
    #     for j in range(width):
            # dst[i,j] = 255-gray[i,j]
    # cv2.imwrite("bar_image.jpg",img)
    cv2.imshow('featuredetect',image_resize(img, ratio=0.125))

    ret, img_binary = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_OTSU)
    cv2.imwrite("bar_image.jpg",img_binary)
    cv2.imshow('img_binary',image_resize(img_binary, ratio=0.125))

    cv2.waitKey(0)


cong = r'--oem 3 --psm 6 outputbase digits'
if __name__ == '__main__':
    image = cv2.imread("imageOCR/WechatIMG980.jpeg")
    featuredetect(image)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    binary = cv2.bitwise_not(image)
    cv2.imshow('binary',image_resize(binary, ratio=0.25))
    #img = cv2.Canny(image, 200, 300)
    #cv2.imshow("Canny",img)

    #image = image_resize(image, ratio=0.5)
    image = cv2.bilateralFilter(image, 15, 5, 5)
    cv2.imshow("Filter",image)

    (_,thresh) = cv2.threshold(image,190,255,cv2.THRESH_BINARY) #Tim use 128
    cv2.imshow("thresh",image_resize(thresh, ratio=0.25))
    print(pytesseract.image_to_string(thresh, config=cong))
    #image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    #bar_image = detect_bar(image)
    #cv2.imshow("bar",bar_image)
    cv2.imwrite("bar_image.jpg",thresh)
    cv2.waitKey(0)