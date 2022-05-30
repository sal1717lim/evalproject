import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import os
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=Axes3D(fig)
import sys
from tqdm import tqdm
cpt=0
for i in tqdm(os.listdir(sys.argv[1]+"\\image")):
    img = cv.imread(sys.argv[1]+"\\image\\"+i,0)
    hist = cv.calcHist([img],[0],None,[256],[0,256])

    x.bar(left=range(256),height= [x[0] for x in hist],zs=cpt,zdir="y")
    cpt+=1

plt.savefig(sys.argv[2]+".png")