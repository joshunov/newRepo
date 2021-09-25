#- Hard: Create a paint program that allows you to draw on screen based on how many fingers you're holding up.
#from sys import last_value
import cv2
import mediapipe as mp
import time
import numpy as np




last_pt = (0,0)               


def distance_between_points(p1,p2,id,lm):
    try:
        if id == p1:
            id_1x = lm.x
            id_1y = lm.y

            # print(lm.x)
            # cv2.circle(draw,(int(lm.x*512),int(lm.y*512)),1,(0,255,0),thickness=-1)

        if id == p2:
            id_2x = lm.x
            id_2y = lm.y

        dis = np.sqrt((id_1x-id_2x)**2+(id_1y-id_2y)**2)
        print(dis)
        return dis

    except(NameError):
        return 0


draw = np.zeros((512,512,3),dtype='uint8')
cap = cv2.VideoCapture(0) #use webcam

#black screen
#we'll be drawing our hands onto here
black_screen = np.zeros((512,512,3),dtype='uint8')


mpHands = mp.solutions.hands #define the function path to .hands
hands  = mpHands.Hands() #create an object from Hands()
mpDraw = mp.solutions.drawing_utils 

#static_image_mode= False - will sometimes detect or track
#                 = True - will always detect
#max_num_hands

pTime =0 #previous time
cTime =0    #current time

last_pt = (0,0)

while True:
    success, img = cap.read()

    #convert to RGB
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #look for the location of hands in the image
    results = hands.process(imgRGB)

    #check if hand has been detected
    if results.multi_hand_landmarks:

        #draw detected hands
        for handLms in results.multi_hand_landmarks:
            #get hand info
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)

                if id == 8:
            
                # draw
                    if(last_pt == (0,0)):
                        curr_point = (int(lm.x*512),int(lm.y*512))
                        last_pt = curr_point
                        continue

                    curr_point = (int(lm.x*512),int(lm.y*512))
                    cv2.line(draw,last_pt,curr_point,(0,255,0),thickness=2)
                    last_pt = curr_point
                

                
                h,w,c = black_screen.shape #height, width, channels
                cx, cy = int(lm.x*w), int(lm.y*h) #center width, center height
            
                #make the palm a different colour and size
                if id ==0: #first landmark
                    cv2.circle(black_screen, (cx,cy), 10, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(black_screen, handLms,mpHands.HAND_CONNECTIONS)
            #HAND_CONNECTIONS returns the hands with connecting lines

    #determine fps (optional)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
    (255,0,255),3)
    
    cv2.imshow("Image", img)
    cv2.imshow("Image", draw)

    cv2.imshow("Black screen",black_screen)

    #we need to reinitialise the black screen after every frame so we can redraw the finger positions
    black_screen = np.zeros((512,512,3),dtype='uint8')

    cv2.waitKey(1)

    #if the d key is pressed, kill screen
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
