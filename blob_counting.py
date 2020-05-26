import cv2
import numpy as np
#load image

image=cv2.imread('image/circle1.jpg',0)
cv2.imshow('Original Image',image)
cv2.waitKey()

#intialize the detector using the default perameter
detector=cv2.SimpleBlobDetector_create()

#detect blobs
keypoints=detector.detect(image)

#Draw blobs on ur image as red circles
blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(image,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs=len(keypoints)
text='Total number of blobs:'+str(number_of_blobs)
cv2.putText(blobs,text,(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(100,0,255),2)

#display image with blob drawKeypoint
cv2.imshow('blob using deafult parameters',blobs)
cv2.waitKey()
#cv2.destroyAllWindows()

#intialize parameters setting using cv2.SimpleBlobDetector
params=cv2.SimpleBlobDetector_Params()
#set area filtering parameters
params.filterByCircularity=True
params.minArea=100

#Set Circularity filtering parameters
params.filterByCircularity=True
params.minCircularity=0.8

#set convexity filtering parameters
params.filterByConvexity=False
params.minConvexity=0.2

#set inertia filtering parameters
params.filterByInertia=True
params.minInertiaRatio=0.02

#create a detector with the parametrers
detector=cv2.SimpleBlobDetector_create(params)

#Detect blobs
keypoints=detector.detect(image)

#Draw blobs on our image as red circles
blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(image,keypoints,blank,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs=len(keypoints)
text='Number of Circuler blobs:'+str(number_of_blobs)
cv2.putText(blobs,text,(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,255),1)
cv2.imshow('filtering circuler blobs only',blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
