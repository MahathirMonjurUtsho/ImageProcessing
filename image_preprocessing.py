import cv2
import matplotlib.pyplot as plt
import numpy as np
file_name = '2.jpg'
img =cv2.imread(file_name,-1)

opt_img = cv2.imread(file_name,-1)
gray_img = cv2.cvtColor(opt_img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray_img,(7,7),2)


#different operators
retval, adaptiveThreshold=cv2.threshold(gray_img,30,255,cv2.THRESH_BINARY)
edges=cv2.Canny(blur,30,30)
laplacian=cv2.Laplacian(img,cv2.CV_64F)
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)


circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,100,param1=2,param2=1,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

max_radius = 0
index = 0
max_index = 0
for i in circles[0,:]:
    if i[2] >= max_radius:
        max_radius = i[2]
        max_index = index
    index = index + 1
    
    
        
cv2.circle(opt_img,(circles[0,max_index][0],circles[0,max_index][1]),circles[0,max_index][2]-2,(255,255,255),-1)
opt_img=cv2.cvtColor(opt_img,cv2.COLOR_BGR2GRAY)
retval, threshold=cv2.threshold(opt_img,254,255,cv2.THRESH_BINARY)

cropped_img = cv2.imread(file_name,-1)
cropped_img = cv2.bitwise_and(cropped_img,cropped_img,mask=threshold)


##rows,cols,channels=cropped_img.shape
##for i in range (0,rows):
##    for j in range (0,cols):
##        px = cropped_img[i,j]
##        if px[2]<50 and px[1]<50 and px[0]<50:
##            cropped_img[i,j] = [0,0,0]
##        if px[2]>80 and px[1]>80 and px[0]>80:
##            cropped_img[i,j] = [0,0,0]
        




cropped_gray = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
img=cv2.bitwise_and(img,img,mask=adaptiveThreshold)
cv2.imshow('original',img)
cv2.imshow('gray image',gray_img)
cv2.imshow('detected circles',opt_img)
cv2.imshow('blurred',blur)
cv2.imshow('mask',threshold)
cv2.imshow('Fundus',cropped_img)
cv2.imshow('adaptiveThreshold',adaptiveThreshold)

##cv2.imshow('edges',edges)
##laplacian=cv2.Laplacian(img,cv2.CV_64F)
##sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
##sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
##cv2.imshow('sobelx',sobelx)
##cv2.imshow('sobely',sobely)
##cv2.imshow('laplacian',laplacian)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(cropped_gray)
ret,adapt = cv2.threshold(cl1,0,30,cv2.THRESH_BINARY)
edges=cv2.Canny(adapt,1,1)
cv2.imshow('clahe',cl1)
##cv2.imshow('edges',edges)

plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
