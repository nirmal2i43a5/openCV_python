# https://www.youtube.com/watch?v=PmZ29Vta7Vc&list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS&index=6

import cv2
import numpy as np
import pickle


# https://docs.opencv.org/4.4.0/d1/de5/classcv_1_1CascadeClassifier.html
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")


labels = {"person_name": 1}


with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	# print(og_labels)  #{'billgates': 0, 'jeff_bezos': 1}
 
	labels = {v:k for k,v in og_labels.items()}
 
cap = cv2.VideoCapture(0)#it captures webcam


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#faces always accept gray
	# print("----",gray)
	
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

	for (x, y, w, h) in faces:#iterating through faces
	#  https://stackoverflow.com/questions/57068928/opencv-rect-conventions-what-is-x-y-width-height
		# print(x, y, w, h)
		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end) #this image is save to item.png
		
		roi_color = frame[y:y+h, x:x+w]#for color images

		# recognize? deep learned model predict keras tensorflow pytorch scikit learn
		id_,conf = recognizer.predict(roi_gray)#run prediction on recognizors
  
		if conf>45:# and conf <= 85:
			# print(id_)
			print(labels[id_])
   
			font  = cv2.FONT_HERSHEY_COMPLEX
			name = labels[id_]
   
			color = [255,255,255]
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)#	cv2.putText(frame,name,(x,y)(is location),font,1(is fontsize),color,stroke,cv2.LINE_AA)
  
  
		img_item = "item.png"
		cv2.imwrite(img_item, roi_color)#taken image is copy to item.png making new folder automatically
  
		color = (255, 0, 0) #BGR 0-255 
		stroke = 2#how thick line i want to make
		end_cord_x = x + w
		end_cord_y = y + h
  
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)#drawing  rectangle blue shape in frame in frame  frame
		# eyes = eye_cascade.detectMultiScale(roi_gray)		
		smile = smile_cascade.detectMultiScale(roi_gray)	
  
  
		for (ex,ey,ew,eh) in smile:
			cv2.rectangle(roi_color, (ex, ey),(ex+ew,ey+eh),(0,255,0),2)
			
  
  
  
  
	cv2.imshow('frame',frame)
   
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
