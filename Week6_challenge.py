'''
## Lesson 3 - Challenge
- Consider `Faces/usrc_all.png` and `Faces/usrc_cropped.png`. 
Using the face detection in part 1, tune the face detection so that all the faces 
in the cropped version are accounted for and there are no extra faces in the uncropped 
version. (Except for the clock. Apparently the clock looks like a face.)

- For each detected face (including the clock, I guess) use the imageFrame in 
`Faces/imageframe.png` to enframe the face, putting the resultant images into 
an output folder.

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img =cv2.imread('Faces/usrc_cropped.png')
cv2.imshow('Lady',img)

#convert to greyscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Lady',gray)

#store the haar face database to haarCascade
haarCascade=cv2.CascadeClassifier('Lesson 3 -Faces/haar_face.xml')

#detect a face and return the rectangular coordinates of the face
facesRect=haarCascade.detectMultiScale(gray,1.3,4) 
                                 
#modify minNeighbors to help filter noise

print(f'Number of faces found = {len(facesRect)}')

files = []
DIR=r'Faces/usrc_faces'
#create list of names by looping through names of folders
for i in os.listdir(DIR):
    files.append(i)

i = 0
for (x,y,w,h) in facesRect:
    cv2.imwrite('Faces/usrc_faces/' + files[i],img[y:y+h , x:x+w])
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    i+= 1
    
    
cv2.imshow('Detected Face',img)
cv2.waitKey(0)
