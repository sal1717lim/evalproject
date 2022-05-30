import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
img_array=[]
for img in tqdm(os.listdir(r"C:\Users\SALIM\PycharmProjects\evalproject\siftdiceL1")):
    data=np.zeros((256*2,256*4,3))
    ssim=cv.imread(r"C:\Users\SALIM\PycharmProjects\evalproject\siftdiceL1\\"+img)
    hsv=cv.imread(r"C:\Users\SALIM\PycharmProjects\evalproject\siftdicessim\\"+img)

    rgb=cv.imread(r"C:\Users\SALIM\PycharmProjects\evalproject\siftgmsdl1\\"+img)
    tir=cv.imread(r"C:\Users\SALIM\PycharmProjects\evalproject\siftgmsdssim\\"+img)

    data[0:256,0:256*2,:]=ssim
    data[256:256*2,0:512,:]=hsv
    data[0:256,256*2:256*4,:]=rgb

    data[256:256*2,256*2:,:]=tir
    cv.putText(img=data, text='diceL1', org=(200, 230), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=3)
    cv.putText(img=data, text='dicessim', org=(160, 512-26), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1,
               color=(255, 255, 255), thickness=3)
    cv.putText(img=data, text='gmsdL1', org=(200+512, 230), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1,
               color=(255, 255, 255), thickness=3)
    cv.putText(img=data, text='gmsdssim', org=(160+512, 512 - 26), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1,
               color=(255, 255, 255), thickness=3)

    cv.imwrite("generated data\\"+img,data)