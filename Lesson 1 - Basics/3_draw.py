#drawing onto images and adding text
import cv2
import numpy as np

blank =np.zeros((500,500,3),dtype='uint8')
#define (height,width,number of colour channels)
#dtype is data type unsigned integer 8
#cv2.imshow('Blank',blank)

#paint the image a cetain colour
blank[:]=0,255,0 #green
#cv2.imshow('Green',blank)

blank[200:300,300:400]=255,0,0 #smaller blue square
#cv2.imshow('Blue',blank)

#draw rectangle
cv2.rectangle(blank,(0,0),(250,250),(0,0,255),thickness=cv2.FILLED)
#cv2.FILLED fills in your shape
#thickness=-1 also works
cv2.imshow('Rectangle',blank)

#draw circle
cv2.circle(blank,(250,250),40,(255,0,0),thickness=2)
cv2.imshow('Circle',blank)

#draw line
cv2.line(blank,(0,0),(250,250),(255,255,0),thickness=3)
cv2.imshow('Line',blank)

#write text
cv2.putText(blank,'Hello World',(255,255),cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.0,color=(255,0,255),thickness=2)
cv2.imshow('Text',blank)


cv2.waitKey(0)