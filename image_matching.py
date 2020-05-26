import cv2
import numpy as np

image=cv2.imread('image/friends1.jpg')
print(image.shape)
cv2.imshow('where is my image',image)
cv2.waitKey()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


template=cv2.imread('image/friends1_test.jpg',cv2.IMREAD_GRAYSCALE)
print(template.shape)
w, h = template.shape[::-1]

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(result)
#create Bounding box
top_left=max_loc
bottom_right=(top_left[0]+50,top_left[1]+50)
cv2.rectangle(image,top_left,bottom_right,(255,0,0),2)

#create Bounding Box

cv2.imshow('where is waldo?',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
