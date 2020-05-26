import cv2
import numpy as np
import math
image=cv2.imread('image/suduko1.jpg')
cv2.imshow('image',image)
cv2.waitKey()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('edges',edges)
cv2.waitKey()
#print(edges)
#run Houghlines using a rho accuracy of 1 pixel
#theta accuracy of np.pi/180 which is 1 degree
#our line threshold is set to 240 (number of points on line)

lines = cv2.HoughLines(edges,1,np.pi/180, 150,None,0,0)
print(len(lines))
#print(lines[0])
#we iterate through each line and convert it to the formet
#required by cv2.lines(i.e requering end points)
if lines is not None:
    for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)

cv2.imshow('Hough Lines',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
