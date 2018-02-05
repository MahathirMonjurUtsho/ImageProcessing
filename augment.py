import cv2
import numpy as np
import random
import sys
import os
import glob

from numpy.random import seed
seed(1)

path = 'image'
outdirpath = 'augmented_image'
if not os.path.exists(outdirpath):
    os.makedirs(outdirpath)


for r1,d,files in os.walk(path):
    for filename in files:
        p = path+'\\'+filename
        image = cv2.imread(p)

        row=image.shape[0]
        col=image.shape[1]

##    outfile = filename
##    outfilepath = os.path.join(outdirpath, outfile)
##    cv2.imwrite(outfilepath, image)

        q = filename.split(".")
        q=q[0]

        ang=random.randint(5,25)
        M=cv2.getRotationMatrix2D((col/2,row/2),ang,1)
        dst=cv2.warpAffine(image,M,(col,row))
        outfile=q+'_1.jpg'
        outfilepath=os.path.join(outdirpath,outfile)
        cv2.imwrite(outfilepath,dst)


        ang = -random.randint(5, 25)
        M = cv2.getRotationMatrix2D((col / 2, row / 2), ang, 1)
        dst = cv2.warpAffine(image, M, (col, row))
        outfile=q+'_2.jpg'
        outfilepath = os.path.join(outdirpath, outfile)
        cv2.imwrite(outfilepath, dst)

        dx=random.randint(5,30)
        dy=0
        dst=cv2.resize(image,(col+dy,row+dx))
        outfile=q+'_3.jpg'
        outfilepath = os.path.join(outdirpath, outfile)
        cv2.imwrite(outfilepath, dst)

        dx = -random.randint(5,30)
        dy = 0
        dst = cv2.resize(image, (col + dy, row + dx))
        outfile=q+'_4.jpg'
        outfilepath = os.path.join(outdirpath, outfile)
        cv2.imwrite(outfilepath, dst)

        dy = random.randint(5,30)
        dx = 0
        dst = cv2.resize(image, (col + dy, row + dx))
        outfile=q+'_5.jpg'
        outfilepath = os.path.join(outdirpath, outfile)
        cv2.imwrite(outfilepath, dst)


        dy = -random.randint(5,30)
        dx = 0
        dst = cv2.resize(image, (col + dy, row + dx))
        outfile=q+'_6.jpg'
        outfilepath = os.path.join(outdirpath, outfile)
        cv2.imwrite(outfilepath, dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

