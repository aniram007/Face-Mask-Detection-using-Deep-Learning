# Face-Mask-Detection-using-Deep-Learning
Trained a custom CNN model to predict if a person is wearing a mask or not by analyzing webcam feed.
The process flow is as follows:
1) Prepare a dataset
  
      1a)For the dataset,I require only the face part and want the rest of the objects in the webcam feed to be cropped out. For this, I use a pre-trained deep-learning-based face detection model (The .prototxt and .caffemodel file and FACE_DET.py contains a function which deploys this model). 
  
      1b)Capture a frame containing your face with either mask/no mask, the captured frame is then cropped so that it contains only the face. This is done with the help of the pre-trained model mentioned in point 1a.The captured frames are then stored in 2 folders: "Mask" and "No Mask" corresponding to facial images with mask and without mask respectively. (The data_collect.py code does this work for you)
  
      1c)Convert the collected dataset into a numpy file for easier processing for training the model. (This is done in FACE_DATA_PREP.py)
      
  
2) Define a custom CNN architecture and train the CNN model on the prepared dataset. (This is done in FACE_TRAIN.py)
  
  
3) Test the performance of the CNN on a live webcam feed. (This is done in FACE_TEST.py)
  
