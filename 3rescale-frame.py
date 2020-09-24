import numpy as np
import cv2

cap = cv2.VideoCapture(0)#it captures webcam


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=30)   #this is 30% of my camera image
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    frame2 = rescale_frame(frame, percent=190)  #it set the difference with above frame
    cv2.imshow('frame2',frame2) #i can have as many frame as i want
    # cv2.imshow('gray',gray)#changing the color of the frame
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()