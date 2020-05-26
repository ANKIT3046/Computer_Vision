import numpy as np
import cv2 as cv

cap=cv.VideoCapture('C:\\Users\DELL\\open_cv\\Background_subtration\\vtest.avi')
#cap=cv.VideoCapture(0)
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()
fgbg2=cv.createBackgroundSubtractorMOG2()


while True:
    ret,frame=cap.read()

    if frame is None:
        break

    fgmask=fgbg.apply(frame)
    fgmask2=fgbg2.apply(frame)

    cv.imshow('Frame',frame)
    cv.imshow('FG Mask Frame',fgmask)
    cv.imshow('FG Mask with shadow',fgmask2)

    keyboard=cv.waitKey(30)

    if keyboard=='q' or keyboard==27:
        break

cap.release()
cv.destroyAllWindows()
