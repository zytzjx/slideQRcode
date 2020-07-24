import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('./imageOCR/IMG_1740.JPG', 0)
img2 = cv2.imread('./imageOCR/1730.png', 0)

# 使用ORB特征检测器和描述符，计算关键点和描述符
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)

# knnMatch 函数参数k是返回符合匹配的个数，暴力匹配match只返回最佳匹配结果。
matches = bf.knnMatch(des1,des2,k=1)

# 使用plt将两个图像的第一个匹配结果显示出来
# 若使用knnMatch进行匹配，则需要使用drawMatchesKnn函数将结果显示
img3 = cv2.drawMatchesKnn(img1=img1,keypoints1=kp1,
                          img2=img2,keypoints2=kp2,
                          matches1to2=matches[:20],
                          outImg=img2, flags=2)
plt.imshow(img3)
plt.show()
# 结果与上图无异，这里不展示了。



# # Load the image
# image1 = cv2.imread('./imageOCR/IMG980.png')

# # Convert the training image to RGB
# training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# # Convert the training image to gray scale
# training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

# # # Create test image by adding Scale Invariance and Rotational Invariance
# # test_image1 = cv2.pyrDown(training_image)
# # test_image = cv2.pyrDown(test_image1)
# # # Display traning image and testing image
# # fx, plots = plt.subplots(1, 2, figsize=(20,10))

# # plots[0].set_title("test_image1")
# # plots[0].imshow(test_image1)

# # plots[1].set_title("test_image")
# # plots[1].imshow(test_image)


# # num_rows, num_cols = test_image.shape[:2]

# # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
# # test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))

# # test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
# test = cv2.imread('./imageOCR/IMG_1740.JPG')
# test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
# test_image = test

# # Display traning image and testing image
# fx, plots = plt.subplots(1, 2, figsize=(20,10))

# plots[0].set_title("Training Image")
# plots[0].imshow(training_image)

# plots[1].set_title("Testing Image")
# plots[1].imshow(test_image)


# orb = cv2.ORB_create()

# train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
# test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

# keypoints_without_size = np.copy(training_image)
# keypoints_with_size = np.copy(training_image)

# cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

# cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Display image with and without keypoints size
# fx, plots = plt.subplots(1, 2, figsize=(20,10))

# plots[0].set_title("Train keypoints With Size")
# plots[0].imshow(keypoints_with_size, cmap='gray')

# plots[1].set_title("Train keypoints Without Size")
# plots[1].imshow(keypoints_without_size, cmap='gray')

# # Print the number of keypoints detected in the training image
# print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

# # Print the number of keypoints detected in the query image
# print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))


# # Create a Brute Force Matcher object.
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# # Perform the matching between the ORB descriptors of the training image and the test image
# matches = bf.match(train_descriptor, test_descriptor)

# # The matches with shorter distance are the ones we want.
# matches = sorted(matches, key = lambda x : x.distance)

# result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)

# # Display the best matching points
# plt.rcParams['figure.figsize'] = [14.0, 7.0]
# plt.title('Best Matching Points')
# plt.imshow(result)
# plt.show()

# # Print total number of matching points between the training and query images
# print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
