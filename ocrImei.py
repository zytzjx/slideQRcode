import argparse
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
'''
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('QRCodeImage/OCR_sample.JPG')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)

'''


# construct the argument parser and parse the arguments (using -im instead image)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
args = vars(ap.parse_args())


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text = pytesseract.image_to_string(Image.open(filename))
    return text


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    print(pytesseract.image_to_string(dst))
    resize = ResizeWithAspectRatio(dst)
    cv2.imshow("custom_blur_demo", resize)

def extrace_object(filename):
    src = cv2.imread(filename)
    #custom_blur_demo(src)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    print(hsv[1242, 627])
    lower_hsv = np.array([160, 20, 100])
    upper_hsv = np.array([175, 150, 255])
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = 0, 0, 0, 0
    if len(contours)>0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

 

    dst = cv2.bitwise_and(src, src, mask=mask)
    cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
    cropimag = src[y:y+h, x:x+w]

    cv2.imwrite("crop.png", cropimag)
    cv2.imshow("crop", cropimag)
    #resize = ResizeWithAspectRatio(dst, height=640)
    #cv2.imshow("mask", resize)
    return cropimag


filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "QRCodeImage/OCR_sample.JPG")

cropimage = extrace_object(filename)
custom_blur_demo(cropimage)
fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),"crop_1.png")

image = cv2.imread(fn)
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
print(pytesseract.image_to_string(image))
cv2.imshow('sharpen_1 Image',output_1)
print(pytesseract.image_to_string(output_1))
cv2.imshow('sharpen_2 Image',output_2)
print(pytesseract.image_to_string(output_2))
cv2.imshow('sharpen_3 Image',output_3)
print(pytesseract.image_to_string(output_3))
#停顿
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()




image = cv2.imread(filename)
resize = ResizeWithAspectRatio(image, height=640) # Resize by width OR
# resize = ResizeWithAspectRatio(image, height=1280) # Resize by height 

cv2.imshow('resize', resize)
cv2.waitKey()