#/bin/env python3

import cv2,sys
import numpy as np

if not len(sys.argv) == 2:
    print("Wrong number of arguments")
    sys.exit(1)

filename = sys.argv[1]

image = cv2.imread(filename)
if 0 == len(image):
    print("couldn't load image.")
    sys.exit(1)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image,(5,5),0)
image2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV ,15,5)

contours, hierarchy = cv2.findContours(image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggest = None
max_area = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 100:
        peri = cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,0.02*peri,True)
        if area > max_area and len(approx)==4:
            biggest = approx
            max_area = area

def dist_line(p1,p2,p3):
    x1,y1 = p1[0]
    x2,y2 = p2[0]
    x0,y0 = p3[0]
    return ((y2 - y1) * x0 - (x2 - x1) * y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2 - x1) ** 2)

def fix_poly(polygon):
    ret = np.array([ [0,0],[0,0],[0,0],[0,0] ],np.float32)
    min_ = np.sqrt(polygon[0][0][0]**2 + polygon[0][0][1]**2)
    minc = 0
    for i in range(1,4):
        if np.sqrt(polygon[i][0][0]**2 + polygon[i][0][1]**2) < min_:
            min_ = np.sqrt(polygon[i][0][0]**2 + polygon[i][0][1]**2)
            minc = i

    #found top left vertex, rotate until it's on the top left
    for i in range(minc):
        polygon = np.roll(polygon,-1,axis=0)

    #if needed, "invert" the order.
    dist1 = dist_line(polygon[0],polygon[2],polygon[1])
    dist3 = dist_line(polygon[0],polygon[2],polygon[3])
    if dist3 > dist1:
        x = polygon[3][0][0]
        y = polygon[3][0][1]
        polygon[3][0][0] = polygon[1][0][0]
        polygon[3][0][1] = polygon[1][0][1]
        polygon[1][0][0] = x
        polygon[1][0][1] = y
    ret[0] = polygon[0][0]
    ret[1] = polygon[1][0]
    ret[2] = polygon[2][0]
    ret[3] = polygon[3][0]
    return ret

dst_bounds = np.array([ [0,0],[499,0],[499,499],[0,499] ],np.float32)
biggest = fix_poly(biggest)	# we put the corners of biggest square in CW order to match with h

transform = cv2.getPerspectiveTransform(biggest,dst_bounds)	# apply perspective transformation
warp = cv2.warpPerspective(image,transform,(500,500))

cv2.imshow("blobs",warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
