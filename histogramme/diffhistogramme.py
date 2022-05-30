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
for i in tqdm(os.listdir(sys.argv[1]+"/image")):
    img = cv.imread(sys.argv[1]+"/image/"+i,0)
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    hist=[x[0] for x in hist]
    reel=cv.imread(sys.argv[3]+"/"+i,0)
    histr=cv.calcHist([reel],[0],None,[256],[0,256])
    histr = [x[0] for x in histr]
    h=[abs(hist[i]-histr[i]) for i in range(len(hist))]
    x.bar(left=range(256),height= h,zs=cpt,zdir="y")
    cpt+=1

plt.savefig(sys.argv[2]+"diif.png")