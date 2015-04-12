#/bin/env python3

import cv2,sys
import numpy as np
from PIL import Image
from pytesseract import image_to_string
if not len(sys.argv) == 2:
    print("Wrong number of arguments")
    sys.exit(1)

filename = sys.argv[1]

image = cv2.imread(filename)
if 0 == len(image):
    print("couldn't load image.")
    sys.exit(1)

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

def to_binary(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image,(5,5),0)
    image2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV ,75,5)
    return image2

def get_tile(image):
    image2 = to_binary(image)

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


    dst_bounds = np.array([ [0,0],[899,0],[899,899],[0,899] ],np.float32)
    biggest = fix_poly(biggest)	# we put the corners of biggest square in CW order to match with h

    transform = cv2.getPerspectiveTransform(biggest,dst_bounds)	# apply perspective transformation
    warp = cv2.warpPerspective(image,transform,(900,900))
    return warp

warp = get_tile(image)

bw = to_binary(warp)
#
# cv2.imshow("bw",bw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sudoku = []
for i in range(9):
    ln = []
    for j in range(9):
        ln.append(bw[i*100:(i+1)*100, j*100:(j+1)*100 ])
    sudoku.append(ln)

def train(): #from opencv tutorial
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,250)[:,np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    return knn

def avg_distance(point,contour):
    dists = [ np.sqrt((point[0] - i[0][0])**2 +(point[1] - i[0][1])**2  ) for i in contour]
    return sum(dists) / len(dists)

def get_number(image):
    image2 = image.copy()
    contours, hierarchy = cv2.findContours(image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    closest = None
    max_dist = 500 #outside of 100x100 image
    arr = []
    for i in contours:
        area = cv2.contourArea(i)
        dist = avg_distance((50,50),i)
        peri = cv2.arcLength(i,True)
        # dist = cv2.pointPolygonTest(i,(len(image)/2,len(image[0])),True)
        if peri > 100 and dist < max_dist and dist < 40:
            max_dist = dist
            closest = i
    if closest == None:
        return None
    arr.append(closest)
    # dst_bounds = np.array([ [0,0],[19,0],[19,19],[0,19] ],np.float32)
    # biggest = fix_poly(biggest)	# we put the corners of biggest square in CW order to match with h
    #
    # transform = cv2.getPerspectiveTransform(biggest,dst_bounds)	# apply perspective transformation
    # warp = cv2.warpPerspective(image,transform,(20,20))
    return arr

def contour_mask(image,contour):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contour, -1, 255,thickness=-1)
    res = cv2.bitwise_and(image,image,mask = mask)
    return res

knn = train()
nimg = np.zeros((900,900,3), np.uint8)

sudoku_contour = []
for i in range(9):
    ln = []
    for j in range(9):
        im = get_number(sudoku[i][j])
        ln.append(im)
    sudoku_contour.append(ln)

numbers = []
for i in range(9):
    ln = []
    for j in range(9):
        if sudoku_contour[i][j] == None:
            ln.append(0)
        else:
            im = contour_mask(sudoku[i][j],sudoku_contour[i][j])
            result = image_to_string(Image.fromarray(im),config="-psm 6")
            # im = cv2.resize(im,(20,20))
            # (result,_,_,_) = knn.find_nearest(im.reshape(-1,400).astype(np.float32),5)
            ln.append(result)
    print ln
    numbers.append(ln)



#
#
# if False:
#     for i in range(9):
#         cv2.line(bw,(i*100,0),(i*100,899),(0,255,0),2)
#         cv2.line(bw,(0,i*100),(899,i*100),(0,255,0),2)
#         cv2.line(nimg,(i*100,0),(i*100,899),(0,255,0),2)
#         cv2.line(nimg,(0,i*100),(899,i*100),(0,255,0),2)
#
# cv2.imshow("blobs" ,nimg)
# cv2.imshow("orig" ,bw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
