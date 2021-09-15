######################################IMPORTING ESSENTIAL LIBRARIES######################################
from tensorflow import keras 											#Tensorflow's frontend to easily deploy Neural networks.
from tensorflow.keras.layers import Flatten,Dense,Activation,Conv2D,MaxPooling2D 					#Layers and activation functions to be used in the neural network.
import numpy as np 												#Python's vector/matrix processing library.
import cv2 												#
import FACE_DET 												#Face Detection library.
##############################################INITIALISATION###############################################
classes = ["Mask","No Mask"]											#List containing target label names.
cap = cv2.VideoCapture(0)                                                               							#Creates an object to obtain images from laptop webcam.
###########################################LOADING THE MODEL###########################################
#model = keras.models.load_model("Models/Mask/weights-improvement-03-1.00.hdf5") 					#Loading the saved model.
model = keras.models.load_model("Models/Mask/best.hdf5")
############################################TESTING THE MODEL###########################################
while True: 												#
	ret, img = cap.read()                                                               							#Obtain image from camera.
	faces = FACE_DET.detect_faces(img.copy())                                         						#Obtain a list of coordinates of rectangles covering the faces.
	if len(faces):											#If any face is detected.
		for (x1,y1,x2,y2) in faces:                                                             						#For each rectangle covering the faces in the image.
			roi = img.copy()[y1:y2, x1:x2]                                                   					#Cut the region of interest from the whole image			
		test = cv2.resize(roi,(50,50)) 									#Resize the input image.
		test=test/255 										#Normalizing the input.
		prediction = model.predict_classes(test.reshape(1,50,50,3)) 						#Get the prediction.
		if prediction[0] == 0: 										#If the person is wearing a mask.
			cv2.putText(img,classes[prediction[0]],(img.shape[1]-150,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,191,0),2) 	#Write the prediction on the image above the bounding box.
		else:
			cv2.putText(img,classes[prediction[0]],(img.shape[1]-150,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) 	#Write the prediction on the image above the bounding box.
	cv2.imshow("Emotion",img)										#Display the output.
	if cv2.waitKey(1) == 27: 										#When escape key is pressed.
		break 											#Terminate the program.
cv2.destroyAllWindows()											#Close all the opened windows.
cap.release() 												#Close the webcame object.
#########################################################################################################
