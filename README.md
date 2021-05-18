# Face detection using Haar and LBP cascade classifier with openCV and Python (Ubuntu 16.04)

In this project, face detection algorithm is implemented using opencv 3 and python using Haar cascade classifier and LBP(Local Binary Patterns) cascade classifier. Both the classifiers are first trained using the available dataset and then used to detect faces of several images. As these classifiers are trained based on data of human faces, it works for human face detection. Similarly if the same classifiers are trained with other objects like cars, trucks, pedestrians, bicycles, etc.then they can be used inside the front camera generally mounted on self driving cars to classify the surrounding objects in real time. 

Hence the aim of this project is to understand the concept and implementation approach of such classifiers via face detection application. 

**Disclaimer:** The images used in this project are randomly taken from Google images. I have no intention to hurt any person or community. If any one find the images inappropriate then kindly contact me.


Folder structure:
1. doc
	* Project details.pdf - Project report
  
2. src
	* FaceDetectionUsingHaarCasacde.py - Python source code of Haar cascade 
	* FaceDetectionUsingLBPcascade.py - Python source code of LBP cascade
	* haarcascade_frontalface_alt.xml - dataset to train Haar cascade classifier (taken from opencv installed folder)
	* lbpcascade_frontalface.xml - dataset to train LBP cascade classifier (taken from opencv installed folder)
	* baby01.jpg, people01.jpg, people02.jpg - images used in project 

3. test
  	* Haar_cascade_babyFace.png, Haar_cascade_people01.png, Haar_cascade_people02.png - haar cascade classifier results
	* LBP_cascade_babyFace.png, LBP_cascade_people01.png, LBP_cascade_people02.png - LBP cascade classifier results
  
