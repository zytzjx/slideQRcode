import cv2
import numpy as np
import pytesseract

cong = r'--oem 3 --psm 6 outputbase digits'

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

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

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def Img_outline1(original_img):
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
    #cv2.imshow("GaussianBlur", blurred)
    _, RedThresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
    return original_img, gray_img, RedThresh, closed, opened


def Img_Outline(input_dir):
    original_img = cv2.imread(input_dir)
    #original_img = image_resize(original_img, height=640)
    return Img_outline1(original_img)

def findContours_img(original_img, opened, bsave):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len (contours) < 2:
        return None, None, 0
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]          # 计算最大轮廓的旋转包围盒
    #print(c)
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    print("angle",angle)
    print("rect", rect)
    box = np.int0(cv2.boxPoints(rect))
    print("box", box)
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 1)
    x,y,w,h = cv2.boundingRect(box)
    cv2.rectangle(draw_img,(x,y),(x+w,y+h),(0,255,0),2)
    crop_img = original_img[y:y+h, x:x+w]
    #cv2.imwrite("imageOCR/1740.png", crop_img)
    rows, cols = original_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    result_img = cv2.warpAffine(original_img, M, (cols, rows))
    return result_img,draw_img, angle


def FindRect(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,11,2)
    print(pytesseract.image_to_string(th1, config=cong))
    cv2.imshow("binary", image_resize(th1,height=640))
    cv2.imshow("mean", image_resize(th2,height=640))
    cv2.imshow("Gaussian", image_resize(th3,height=640))

if __name__ == "__main__":
    input_dir = "/home/qa/Desktop/slideqrcode/bar_image.jpg"
    # FindRect(input_dir)
    original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
    original_img = cv2.medianBlur(original_img,  5)
    ret,th1 = cv2.threshold(original_img, 130, 255, cv2.THRESH_BINARY)
    print(pytesseract.image_to_string(th1, config=cong))

    result_img,draw_img,angle = findContours_img(original_img,opened,False)

    # cv2.imshow("original_img", original_img)
    # cv2.imshow("gray_img", gray_img)
    # cv2.imshow("RedThresh", RedThresh)
    # cv2.imshow("Close", closed)
    # cv2.imshow("Open", opened)
    # cv2.imshow("draw_img", draw_img)
    # cv2.imshow("result_img", result_img)

    original_img = cv2.imread(input_dir)
    rows, cols = original_img.shape[:2]
    print("imagesize:",rows, "X",cols)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    result_img = cv2.warpAffine(original_img, M, (cols, rows))
    rows, cols = result_img.shape[:2]
    print("imagesizeq:",rows, "X",cols)
    # result_img = cv2.resize(result_img, (500, 500))
    # cv2.imshow("originalrotate", result_img)
    # original_img, gray_img, RedThresh, closed, opened =Img_outline1(result_img)
    # result_img,draw_img,angle = findContours_img(original_img,opened,True)
    result_img = cv2.medianBlur(result_img,  5)
    ret,th1 = cv2.threshold(result_img, 110, 255, cv2.THRESH_BINARY)
    print(pytesseract.image_to_string(th1, config=cong))
    #draw_img = cv2.resize(draw_img, (500, 500))
    cv2.imshow("draw_img", th1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
