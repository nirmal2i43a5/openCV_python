import numpy as np
import cv2

cap = cv2.VideoCapture(0)#it captures webcam


def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)
    
make_720p()#changes entire video capture resolution
change_res(4000, 2000)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operation on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    # cv2.imshow('frame1',frame) #i can have as many frame as i want
    cv2.imshow('gray',gray)#changing the color of the frame
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()