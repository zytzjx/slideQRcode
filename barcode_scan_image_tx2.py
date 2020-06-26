# import the necessary packages
from pyzbar import pyzbar
import argparse
import cv2
import imutils
import numpy as np
import re
import math

 
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

def decodeBarcode(image):
    imaget = image_resize(image, ratio=1.0)#cv2.imread('crop.png')
    for angle in np.arange(0, 45, 1):
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
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 255), 1)

            # print the barcode type and data to the terminal
            print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
            
            if isValidEMEI("{}".format(barcodeData)): 
                return "{}".format(barcodeData)
            
        #cv2.imshow("Rotated (Correct)", rotated)
        #cv2.waitKey(0)

        if len(barcodes) > 0:
            break


# construct the argument parser and parse the arguments (using -im instead image)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])

print("find:", decodeBarcode(image))

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

