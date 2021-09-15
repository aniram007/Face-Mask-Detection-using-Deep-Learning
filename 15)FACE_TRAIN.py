######################################IMPORTING ESSENTIAL LIBRARIES######################################
from tensorflow import keras 											#Tensorflow's frontend to easily deploy Neural networks.
from tensorflow.keras.layers import Flatten,Dense,Activation,Conv2D,MaxPool2D,Dropout 					#Layers and activation functions to be used in the neural network.
from tensorflow.keras.callbacks import ModelCheckpoint 								#This is used for checkpointing the model.
import numpy as np 												#Python's vector/matrix processing library.
import cv2 												#
#########################################DATASET AUGMENTATION#########################################
def augment(img): 												#Augmentation function.
  M=cv2.getRotationMatrix2D((25,25),np.random.randint(-10,10),1)							#Matrix to randomly rotate the image anywhere by -10 to 10 degreees about the center.
  img=cv2.warpAffine(img,M,(50,50)) 										#Rotating the image using the matrix.
  return img 												#Return the augmented image.
##############################################TRAINING DATA##############################################
dataset = np.load("Datasets/MASK.npy",allow_pickle = True)							  	#Cat Dog dataset.
X = [] 													#Training inputs.
Y = [] 													#Training Targets.
for image,label in dataset: 											#Loop to store inputs and targets separately. The last 1000 values are testing values.
	X.append(augment(image)) 										#Append the current input into the input array after augmentation.
	Y.append(label) 											#Append the current target into the target array.
X = np.array(X).reshape(-1,50,50,3) 										#Converting the input list to a numpy array.
Y = np.array(Y) 												#Converting the target list to a numpy array.
X = X/255.0 												#Normalizing the inputs.
cv2.imshow("Training Data example",X[0]) 									#Display an input image.
print(Y[0]) 													#Print the target value of the training input.
cv2.waitKey(0) 												#Wait for user interrupt.
cv2.destroyAllWindows()											#Close the image window.
###########################################DEFINING THE MODEL###########################################
model = keras.Sequential() 											#Initialising the model for training.
model.add(Conv2D(32,(3,3),padding='same',input_shape=X.shape[1:])) 							#Convolutional layer with 32 feature maps and 3x3 kernels.
model.add(Activation('relu')) 											#ReLU activation function.
model.add(Dropout(0.3)) 											#Neuron dropout probability is 30%.
model.add(MaxPool2D(pool_size=(2,2))) 									#Maxpooling layer reduces the size of the feature maps by half.
model.add(Conv2D(64,(3,3),padding='same'))									#Convolutional layer with 64 feature maps and 3x3 kernels.
model.add(Activation('relu')) 											#ReLU activation function.
model.add(Dropout(0.3))											#Neuron dropout probability is 30%.
model.add(MaxPool2D(pool_size=(2,2))) 									#Maxpooling layer reduces the size of the feature maps by half.
model.add(Conv2D(128,(3,3),padding='same'))									#Convolutional layer with 32 feature maps and 3x3 kernels.
model.add(Activation('relu')) 											#ReLU activation function.
model.add(Dropout(0.3))											#Neuron dropout probability is 30%.
model.add(MaxPool2D(pool_size=(2,2))) 									#Maxpooling layer reduces the size of the feature maps by half.
model.add(Conv2D(256,(3,3),padding='same'))									#Convolutional layer with 32 feature maps and 3x3 kernels.
model.add(Activation('relu')) 											#ReLU activation function.
model.add(Dropout(0.3))											#Neuron dropout probability is 30%.
model.add(MaxPool2D(pool_size=(2,2))) 									#Maxpooling layer reduces the size of the feature maps by half.
model.add(Flatten()) 											#Flatten the feature maps to a 1-D vector.
model.add(Dense(128)) 											#Fully connected layer with 128 neurons.
model.add(Activation("relu")) 											#ReLU activation function.
model.add(Dropout(0.3)) 											#Neuron dropout probability is 30%.
model.add(Dense(2)) 											#Output neuron to predict animal.
model.add(Activation("softmax")) 										#Sigmoid activation function returns a value between 0-1.
###########################################TRAINING THE MODEL###########################################
model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(),metrics=['accuracy'])			#Using Adam optmiser and mean square error to optimise the model.
filepath="Models/Mask/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5" 						#Path to save the checkpoints.
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') 			#Save the models with the lowest validation loss upto that point.
callbacks_list = [checkpoint] 											#Used in model.fit.
model.fit(X, Y, validation_split=0.20, epochs=30, batch_size=10, callbacks=callbacks_list, verbose=1)				#Train the model for 30 epochs using 30% of the data as validation data.############################################SAVING THE MODEL############################################
#########################################################################################################
