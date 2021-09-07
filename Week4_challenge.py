import cv2
import numpy

#reading images
img = cv2.imread('Photos/Bughead.jpg')
cv2.imshow('Bughead', img)

cv2.resize(img, (1000,1000))

#draw rectangle
rectangle = cv2.rectangle(img,(250,250),(750,500),(40,100,255),thickness=cv2.BORDER_ISOLATED)
cv2.imshow('Rectangle',rectangle)

#greyscale, blur, canny
img = cv2.imread('Photos/Bughead.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT)
img = cv2.Canny(img,200,200)
cv2.imshow('greyscale, blur, canny',img)

#rotate
img = cv2.imread('Photos/Bughead.jpg')

(height,width)=img.shape[:2]
rotPoint=(width//2,height//2)
    
rotMat = cv2.getRotationMatrix2D(rotPoint,45,scale=1.0)
img = cv2.warpAffine(img,rotMat,(width,height))

cv2.imshow('Rotate',img)


cv2.waitKey(0)
cv2.destroyAllWindows()