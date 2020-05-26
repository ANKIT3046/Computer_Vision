import cv2
import numpy as np

def ORB_detector(new_image,image_template):
    #function that compute input image to template
    #it then returs the number of ORB matches between them

    image1=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)

    #create orb detector with 1000 keypoints with a sclaling pyramid factor of 1.2
    orb=cv2.ORB_create(500,1.2)
    #detect keypoints of original image
    (kp1,des1)=orb.detectAndCompute(image1,None)

    #detect keypoints of rotated image
    (kp2,des2)=orb.detectAndCompute(image_template,None)

    ##create matcher
    #note we're no longer using flannbased matching
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    #Do matching
    matches=bf.match(des1,des2)

    #sort the matches based on distence least distance is better
    matches=sorted(matches,key=lambda val:val.distance)

    return len(matches)

cap=cv2.VideoCapture(0)

#load our image template ,this is our refrance image
image_template=cv2.imread('image/abc.jpg',0)

while True:
    #get web cam image
    ret,frame=cap.read()

    #get height and width of webcam frame
    height,width=frame.shape[:2]

    #define ROI Box Dimensions (Note some of these things sould be outside the loop)

    top_left_x=width/3
    top_left_y=(height/2)+(height/4)
    bottom_right_x=(width/3)*2
    bottom_right_y=(height/2)-(height/4)

    #draw rectanguler window for our refion of interest
    cv2.rectangle(frame,(int(top_left_x),int(top_left_y)),(int(bottom_right_x),int(bottom_right_y)),(255,0,0),3)

    #crop window of observation we defined above
    cropped=frame[int(bottom_right_y):int(top_left_y),int(top_left_x):int(bottom_right_x)]

    #flip frame orientation horizontally
    frame=cv2.flip(frame,1)
    #get number of ORB matches
    matches=ORB_detector(cropped,image_template)

    #Display status string showing the current no. of matches
    output_string='Matches= '+str(matches)

    cv2.putText(frame,output_string,(50,450),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,150),2)

    #our threshold to indicate object detection
    #for new image or lightning condition you may need to experiment a bit
    #note the : ORB detector to get the top 1000 matches ,350 is essential a min 35% match
    threshold=100
    #if matches exceed our threshold then our object has been detected
    if matches>threshold:
        cv2.rectangle(frame,(int(top_left_x),int(top_left_y)),(int(bottom_right_x),int(bottom_right_y)),(0,255,0),3)
        cv2.putText(frame,'Object Found',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow('Object detection using ORB',frame)

    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()
