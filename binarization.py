#/bin/env python3

import cv2,sys
if not len(sys.argv) == 2:
    print("Wrong number of arguments")
    sys.exit(1)

filename = sys.argv[1]

image = cv2.imread(filename)
if 0 == len(image):
    print("couldn't load image.")
    sys.exit(1)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV ,7,5)

cv2.imwrite("binarized_" + filename.split('/')[-1],image2)
