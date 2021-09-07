"New python file opening a photo of me with a bughead"
import cv2

bug = cv2.imread('Photos/Bughead.jpg')
cv2.imshow('Bugboi', bug)

cv2.waitKey(0)