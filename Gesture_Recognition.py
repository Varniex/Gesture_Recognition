## @Author: Dishant Varshney
""" Recognising Gestures and Control the screen as well using the libraries OpenCV, Numpy, and pynput
 Currently by this code you can control the space-bar key - lowering down the finger(s) presses the spacebar"""

## Importing Libraries
import numpy as np 
import cv2 as cv 
import math 
from pynput.keyboard import Key, Controller

## Defining constants
lwr = np.array([0,50,70], np.uint8)
upr = np.array([100,230,230], np.uint8)
kernel = np.ones((5,5), np.uint8)
fingers = []  # track the path of finger

kbd = Controller() # Defining the controller
st = False  # press 's' to start
kSp = False

## Functions
# HSV function to extract hand from background
def hsvF(focusF):
    hsv = cv.cvtColor(focusF, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lwr, upr)
    mask = cv.dilate(mask, kernel, iterations = 3)
    mask = cv.GaussianBlur(mask, (5,5), 100)
    return mask

# Centroid function to detect the centroid of the contour
def centroidF(cnt):
    M = cv.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx, cy
    else:
        return None

# function to display the hishest point in the contour
def points(fingers, focusF):
    for i in range(len(fingers)):
        cv.circle(focusF, fingers[i], 4, (0,0,100), -1)

## Main function
# Capture the frames from the web camera
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv.flip(frame, 1)
        focusF = frame[55:345, 305:595]
        frameC = frame[50:300, 350:600]

        cv.rectangle(frame, (300, 50), (600, 350), (200,0,0), 1)
        cv.putText(frame, 'DV', (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        noise = hsvF(focusF)

        r,thresh = cv.threshold(noise, 100, 255, cv.THRESH_BINARY)
        img, cont, hie = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        try:
            cnt = max(cont, key = cv.contourArea)

            epsilon = 0.001*cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)

            hull = cv.convexHull(approx, returnPoints=False)
            area_cnt = cv.contourArea(approx)
            defects = cv.convexityDefects(approx, hull)

            centroid = centroidF(cnt)
            n = 0
            if st == True:
                if area_cnt > 11000 and defects is not None:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(approx[s][0])
                        end = tuple(approx[e][0])
                        far = tuple(approx[f][0])
                        ftop = tuple(cnt[cnt[:,:,1].argmin()][0])

                        if len(fingers) < 10:
                            fingers.append(ftop)
                        else:
                            fingers.pop(0)
                            fingers.append(ftop)
                        points(fingers, focusF)
                        
                        cv.line(focusF, start, end, (0,255,0), 2)
                        cv.circle(focusF, centroid, 3, (0,255,255), -1)

                        a = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
                        b = math.sqrt((far[0] - end[0])**2 + (far[1] - end[1])**2)
                        c = math.sqrt((start[0] - far[0])**2 + (start[1] - far[1])**2)
                        # s = (a+b+c)/2
                        # ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                        # dis = (2*ar)/a

                        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c))

                        if angle < math.pi/2: # and dis > 30:
                            n += 1

                    if centroid[1] - ftop[1] >=90:
                        n += 1
                    n = min(n,5)
                    cv.putText(frame, f"{n}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,150), 2)
                    
                    # defining the gestures to control screen
                    if kSp == True and n == 0:
                        kbd.press(Key.space)
                        kbd.release(Key.space)
                        kSp = not kSp
                    elif kSp == False and n >= 1:
                        kSp = not kSp
                else:
                    cv.putText(frame, "Can't detect anything", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,150), 2) 
            else:
                cv.putText(frame, "Press 's' to start", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,150), 2)
        except:
            cv.putText(frame, "Put your hand in the frame", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        cv.imshow('Finger Detection', frame)

        k = cv.waitKey(1) & 0xff
        if k  == ord('s'):
            st = not st
        elif k == ord('q'):
            break
            
cap.release()
cv.destroyAllWindows()
