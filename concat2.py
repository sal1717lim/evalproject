import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
img_array=[]
for img in tqdm(os.listdir(r"C:\Users\SALIM\PycharmProjects\evalproject\feriel dataset\images")):
    data=np.zeros((256,256*2,3))
    ssim=cv.imread(r"C:\Users\SALIM\PycharmProjects\evalproject\feriel dataset\images\\"+img)
    hsv=cv.imread(r"C:\Users\SALIM\PycharmProjects\evalproject\feriel dataset\predict2\\"+img)


    data[0:256,0:256,:]=ssim
    data[0:256,256:512,:]=hsv


    cv.imwrite("generated data2\\"+img,data)