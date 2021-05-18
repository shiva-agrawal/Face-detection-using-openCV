'''
Developer  : Shiva Agrawal
Date       : 06.10.2018
Version    : 01
Description: LBP (Local Binary Patterns) cascade claasifier based human face detection algorithm using opencv 3
             It uses the readily available lbpcascade_frontalface.xml training dataset
             and then detects the human faces.
'''

# import the opencv library for python
import cv2;

#-------------------------------------------------------------------------------------------------------
## Step 1: Loading and pre-processing of image
#-------------------------------------------------------------------------------------------------------

# load the original color image where face detection is required
# change image name here to load another image for face detection
image =   cv2.imread('people01.jpg', cv2.IMREAD_COLOR)

# display the original image
cv2.namedWindow("Original Image")
cv2.imshow("Original Image",image)

# make a deep copy of the image to do further processing on copied image
image_copy = image.copy()

# As LBP cascade algorithm requires grayscale image to process, the image is here converted
# from RGB to Grayscale using cvtColor function
grayImage = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY);

#--------------------------------------------------------------------------------------------------------
## Step 2:Implementing LBP cascade classifier for face detection
#--------------------------------------------------------------------------------------------------------

# load LBP cascade classifier training file to train the classifier.
# The training file is available in opencv files
lbp_cascade_classifier_training_data = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# detect the faces using the classifier
# scaleFactor = 1.1 and minNeighbours = 5 is selected after some trial and error and using some online references
faces = lbp_cascade_classifier_training_data.detectMultiScale(grayImage,scaleFactor=1.1,minNeighbors=5)

# print total count of detected faces in the image
print('Faces found = ', len(faces))


# faces variable contains the information of each detected face as rectangle values and hence this is used to
# draw actual rectangle on the original Image to highlight the detected / recognized faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#---------------------------------------------------------------------------------------------------------
## step 3: Display the final image
#----------------------------------------------------------------------------------------------------------
cv2.namedWindow("LBP cascade Face Detected Image")
cv2.imshow("LBP cascade Face Detected Image",image )

# hold and destroy the window upon exit
cv2.waitKey(0)
cv2.destroyAllWindows()