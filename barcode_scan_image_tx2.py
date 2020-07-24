# import the necessary packages
from pyzbar import pyzbar
import argparse
import cv2
import imutils
import numpy as np
import re
import math
import os
import time
from glob import glob
from datetime import datetime

 
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


# Python3 code to check whether the 
# given EMEI number is valid or not 
decimal_decoder = lambda s: int(s, 10)
decimal_encoder = lambda i: str(i)


def luhn_sum_mod_base(string, base=10, decoder=decimal_decoder):
    # Adapted from http://en.wikipedia.org/wiki/Luhn_algorithm
    digits = list(map(decoder, string))
    return (sum(digits[::-2]) +
        sum(list(map(lambda d: sum(divmod(2*d, base)), digits[-2::-2])))) % base

# Returns True if n is valid EMEI 
def isValidEMEI(n): 
    p = re.compile('\d{15}')
	# Converting the number into 
	# Sting for finding length 
    s = str(n) 
	# If length is not 15 then IMEI is Invalid 
    if p.match(s) is None: 
        return False

    return luhn_sum_mod_base(s) == 0

def DecodeBarcodes(images):
    for image in images:
        barcodes = pyzbar.decode(image)
        # loop over the detected barcodes
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw the
            # bounding box surrounding the barcode on the image
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # the barcode data is a bytes object so if we want to draw it on
            # our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            # draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, barcodeType)
            # print(text)
            # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5, (255, 0, 255), 1)

            # print the barcode type and data to the terminal
            # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
            
            if isValidEMEI("{}".format(barcodeData)): 
                return "{}".format(barcodeData)


def DecodeRotateBar(image):
    images=[]
    ratios=[0.5, 0.25, 0.75, 1.0]
    for ratio in ratios:
        images.append(image_resize(image, ratio=ratio))

    for angle in np.arange(0, 45, 2):
        imagerotes=[]
        for img in images:
            imagerotes.append(imutils.rotate_bound(img, angle))
        
        imei = DecodeBarcodes(imagerotes)
        if imei is not None and len(imei) > 0:
            return imei


def decodeBarcode(image, ratio = 0.5):
    imaget = image_resize(image, ratio=ratio)#cv2.imread('crop.png')
    bFind = False
    imei=""
    for angle in np.arange(0, 45, 2):
        print("image angle:", angle)
        rotated = imutils.rotate_bound(imaget, angle)
        # find the barcodes in the image and decode each of the barcodes
        barcodes = pyzbar.decode(rotated)

        # loop over the detected barcodes
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw the
            # bounding box surrounding the barcode on the image
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # the barcode data is a bytes object so if we want to draw it on
            # our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            # draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, barcodeType)
            print(text)
            # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5, (255, 0, 255), 1)

            # print the barcode type and data to the terminal
            # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
            
            if isValidEMEI("{}".format(barcodeData)): 
                imei = "{}".format(barcodeData)
                bFind = True
                #return "{}".format(barcodeData)
            
        #cv2.imshow("Rotated (Correct)", rotated)
        #cv2.waitKey(0)

        #if len(barcodes) > 0:
        if bFind:
            break

    return imei
    

'''
imageOCR/IMG_1763.JPG
imageOCR/IMG_1764.JPG
imageOCR/IMG_1768.JPG
imageOCR/IMG_1777.JPG
imageOCR/IMG_1780.JPG
imageOCR/IMG_1816.JPG
imageOCR/IMG_1824.JPG

import os
from glob import glob
result = [y for x in os.walk('/home/qa/Desktop/0721') for y in glob(os.path.join(x[0], '*.jpg'))]
'''
# construct the argument parser and parse the arguments (using -im instead image)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to input image")
args = vars(ap.parse_args())
# args["image"] = "imageOCR/IMG_1824.JPG"

if  args["image"] is not None:
    # load the input image
    image = cv2.imread(args["image"])
    print("find:", decodeBarcode(image))
else:
    #files = glob.glob('imageOCR/IMG_*.JPG')
    files = [y for x in os.walk('/home/qa/Desktop/16412') for y in glob(os.path.join(x[0], '*.jpg'))]
    files.sort()
    '''
    files=["/home/qa/Desktop/0721/94932_A1/2.jpg",
"/home/qa/Desktop/0721/63519_A1/2.jpg",
"/home/qa/Desktop/0721/73117_A1/2.jpg",
"/home/qa/Desktop/0721/77222_A1/2.jpg",
"/home/qa/Desktop/0721/81187_A1/2.jpg",
"/home/qa/Desktop/0721/58180_A1/2.jpg",
"/home/qa/Desktop/0721/51312_A1/2.jpg",
"/home/qa/Desktop/0721/53060_A1/2.jpg",
"/home/qa/Desktop/0721/40395_A1/2.jpg",
"/home/qa/Desktop/0721/16412_A1/2.jpg"]
    '''
    ratios=[0.5, 0.25, 0.75, 1.0]
    print("count=", len(files))
    print("now =", datetime.now().time()) 
    for fn in files:
        print(fn)
        image = cv2.imread(fn)
        #print("find:", decodeBarcode(image))
        bFind = False
        sImei = DecodeRotateBar(image)
        if sImei is not None and len(sImei)>0:
            print(fn, "=", sImei)
            bFind = True
        # for ratio in ratios :
        #     sImei = decodeBarcode(image, ratio)
        #     if sImei is None or len(sImei) == 0:
        #         # sImei = decodeBarcode(image, 0.25)
        #         # if sImei is None or len(sImei) == 0:        
        #         #     print("Decode Fail:", fn)
        #         # else:
        #         #     print(fn, "=", sImei)
        #         pass
        #     else:
        #         print(fn, "=", sImei)
        #         bFind = True
        #         break

        if not bFind :
            print("Decode Fail:", fn)

    print("end =", datetime.now().time()) 

# # find the barcodes in the image and decode each of the barcodes
# barcodes = pyzbar.decode(image)

# # loop over the detected barcodes
# for barcode in barcodes:
# 	# extract the bounding box location of the barcode and draw the
# 	# bounding box surrounding the barcode on the image
# 	(x, y, w, h) = barcode.rect
# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 	# the barcode data is a bytes object so if we want to draw it on
# 	# our output image we need to convert it to a string first
# 	barcodeData = barcode.data.decode("utf-8")
# 	barcodeType = barcode.type

# 	# draw the barcode data and barcode type on the image
# 	text = "{} ({})".format(barcodeData, barcodeType)
# 	print(text)
# 	cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
# 		0.5, (255, 0, 255), 1)

# 	# print the barcode type and data to the terminal
# 	print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

# # show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)

