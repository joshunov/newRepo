#basic functions
import cv2
img = cv2.imread('..\Photos\cat.jpg')
cv2.imshow('Cat', img)

#convert to greyscale
greyscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Grey',greyscale)

#blur
blur=cv2.GaussianBlur(img,(9,9),cv2.BORDER_DEFAULT)
#ksize has to be odd numbers
cv2.imshow('Blur',blur)

#canny
canny=cv2.Canny(img,125,200)
cv2.imshow('Canny',canny)

#dilate
#used to accentuate edge cases
dilated=cv2.dilate(canny,(3,3),iterations=1)
cv2.imshow('Dilated',dilated)

#erode
#used to diminish edge cases
eroded=cv2.erode(dilated,(3,3),iterations=1)
cv2.imshow('Eroded',eroded)

#resize
resized=cv2.resize(img,(500,500))
cv2.imshow('Resize',resized)

cv2.waitKey(0)
