#Importing required packages.
import cv2
import matplotlib.pyplot as plt

#Reading the image and converting it to grayscale.
img = cv2.imread('image_name.jpg')
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plt.imshow(g_img, cmap='gray')

'''cv2.imshow('Test Imag',g_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#Loading the haar cascade file.
haar_face_cascade = cv2.CascadeClassifier('face.xml')

#Most important step as it detects the faces and also tuning is performed here. Keep in mind that values of
#parameters may vary from image to image.

#Detecting faces.
faces = haar_face_cascade.detectMultiScale(g_img,
                                           scaleFactor = 1.1,
                                           minNeighbors = 3,
                                           flags = cv2.CASCADE_SCALE_IMAGE)

#Adding rectangle around the faces detected.
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x, y),(x+w, y+h),(0,255,0),2)

#plt.imshow(img)
cv2.imshow('Testing Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()