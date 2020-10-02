import numpy as np
import cv2

cap = cv2.VideoCapture(0)#it captures webcam

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting fram
    cv2.imshow('frame',frame)
    
    cv2.imshow('frame1',frame) #i can have as many frame as i want
    cv2.imshow('gray',gray)#changing the color of the frame
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()