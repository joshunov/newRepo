'''
- Challenge: Draw the grid lines of the sudoku image by 
detecting the lines using a Hough transform. Use image sudoku.png from the Github.
'''
import cv2
import numpy as np

img = cv2.imread('Photos/sudoku.png')
edge = cv2.Canny(img,100,200)

minLineLength=20
maxLineGap=5
edge = cv2.Canny(img,100,200)
lines = cv2.HoughLinesP(edge,1,np.pi/180,100,minLineLength,maxLineGap)

blank = np.zeros(img.shape)

for l in lines:
    for x1,y1,x2,y2 in l:
        cv2.line(blank,(x1,y1),(x2,y2),(255,255,255),5)

cv2.imshow("SUDOKU", img)
cv2.imshow("lines",blank)
cv2.waitKey(-1)