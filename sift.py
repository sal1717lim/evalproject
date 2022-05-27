import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
def sift(image1,image2):
    img1 = cv.imread(image1,cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread(image2,cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
    sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("sift"+sys.argv[1]+"/"+image1[-14:],img3)
import os
import sys
if not os.path.exists("sift"+sys.argv[1]):
    os.mkdir("sift"+sys.argv[1])
from tqdm import tqdm
for img in tqdm(os.listdir(sys.argv[1]+"/image")):
    sift(sys.argv[1]+"/image/"+img,"original/"+img)