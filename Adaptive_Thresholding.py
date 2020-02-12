import dlib
import cv2
import time
import winsound
import numpy as np
from calib import *
import math

threshold = 0.23
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Total_blinks=0 #Blinks
close_frame=0 # Number of frames for which eyes were closed ,used for calculating one blink
itime=time.time() #initial time
Avg_EAR=0 #The average EAR for the last few duration of time
blinks_per_minute=0 #Blinks per minute
Avg_blinks_per_minute=4.00 #Number of blinks per minute
t=0 #time passed in each cycle
n=1 #number of cycles or minutes
E=0 # Instantanious EAR
M=0 # Instantanious MAR
Earray=np.ones((1,20),dtype=float)*0.25 #Numpy Array for storing the last specified number of calculated EAR values

print("In calib")

print(threshold)
V=cv2.VideoCapture(0)
while(V.isOpened()):

    r,frame= V.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,1)
    (b, l) = gray.shape
    #If no faces have been detected sets Avg values of EAR and MAR to defaults so as to not trigger the alarm
    if len(rects)==0:
        Avg_EAR=0.3
        E=0.3
        M=0.4
    if not detector(gray):
        c = c+1
        print ("nif")

    for (i,rect) in enumerate(rects):#Each rect represents one face
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        (x1, y1) = shape[30]   #nose tip
        (x2, y2) = shape[8]    #chin
        (x3, y3) = shape[36]   #right eye corner
        (x4, y4) = shape[45]   #left eye corner
        imagePoints = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)
        objectPoints = np.array([(0, 0, 0), (0, -150, -50), (100, 75, -50), (-100, 75, -50)], dtype=np.float64)
        cameraMatrix = np.array([[l, 0, l/2], [0, l, b/2], [0, 0, 1]], dtype=np.float64)

        re, ro, tr = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, np.zeros(4, dtype=int), np.zeros([3, 1]), np.zeros([3, 1])
                                  , True, flags=cv2.SOLVEPNP_ITERATIVE)
        #print "rotation vector={}".format(ro)
        # cv2.putText(frame, "q".format(q), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1)

        rotation_matrix, j = cv2.Rodrigues(ro)
        q = math.sqrt(pow(ro[1], 2) + pow(ro[2], 2))
        #cv2.putText(frame, "q={}".format(q), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        print ("q={}".format(q))
        #print "rotation matrix={}".format(rotation_matrix)

        if q >= 0.45:
            c = c+1
        else:
            c = 0
        (x,y,w,h)=rect_to_bb(rect)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),6,8)
        shape=predictor(gray,rect)
        shape=shape_to_np(shape)
        for (x, y) in shape[36:48]:
	        cv2.circle(frame, (x, y),2, (0, 255, 0), -1)
        for (x, y) in shape[48:68]:
	        cv2.circle(frame, (x, y),2, (255, 0, 0), -1)
        #cv2.circle(frame, (shape[43,0],shape[43,1]),2, (0, 255, 255), -1)
        #cv2.circle(frame, (shape[47,0],shape[47,1]),2, (0, 255, 255), -1)

        E=EAR(shape)
        cv2.putText(frame,"Avg EAR={:1.2f}".format(Avg_EAR),(30,30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(frame,"EAR={:1.2f}".format(E),(200,30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        # cv2.putText(frame,)
        #cv2.putText(frame,"EAR={}".format(n),(400,30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        #This section calculates each blink
        if E<0.25:
            close_frame+=1
        elif (close_frame>0):
            Total_blinks+=1
            blinks_per_minute+=1
            close_frame=0
        cv2.putText(frame,"Total blinks={}".format(Total_blinks),(30,50),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(frame,"Blinks per minute={:2.2f}".format(Avg_blinks_per_minute*4),(200,50),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(frame,"Time={:2.0f}".format(t),(420,50),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)


        M=MAR(shape)
        cv2.putText(frame,"MAR={:1.2f}".format(M),(400,30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    if c >= 6:
        cv2.putText(frame, 'WARNING : DISTRACTED DRIVER', (100, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    for i in range(19,0,-1):
        Earray[0,i]=Earray[0,i-1]

    Earray[0,0]=E
    Avg_EAR=np.sum(Earray)/20

    if Avg_EAR<threshold:
        cv2.putText(frame,"SLEEPY EYES",(60,450),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,255),11)
        winsound.Beep(350,200)
    if M>1.5:
        cv2.putText(frame,"Be alert!",(200,250),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),15)
        winsound.Beep(450,200)
    #if Avg_blinks_per_minute>5.25:
        #cv2.putText(frame,"Critical blink rate",(50,350),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),7)

    t=time.time()-itime

    #Here time is being recycled after each cycle
    if t>=14.99:
        Avg_blinks_per_minute=(Avg_blinks_per_minute*n+blinks_per_minute)/(n+1)
        blinks_per_minute=0
        t=0
        n+=1
        itime=time.time()
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)

    #To reset the values of the important variables
    if key==ord('r'):
        t=0
        Total_blinks=0
        Avg_blinks_per_minute=4
        blinks_per_minute=0
        itime=time.time()
        n=1
    if key==ord('q'):
        break
    if key==ord('c'):
        cv2.destroyAllWindows()
        threshold = Calibration()
        print (threshold)
        V=cv2.VideoCapture(0)
V.release()
cv2.destroyAllWindows()
