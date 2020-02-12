import dlib
import cv2
import time
import winsound
import numpy as np
#Function to change the dlib shape object to numpy array
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

#Function to convert the dlib rect containing the face boundary object into cv2 format
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def dis(p1,p2,shape):
    z=((shape[p1,0]-shape[p2,0])**2+(shape[p1,1]-shape[p2,1])**2)**(1/2)
    return z

#EAR and MAR calculations
def EAR(shape):
    return float((((dis(41,37,shape)+dis(38,40,shape))/(2*dis(36,39,shape)))+((dis(43,47,shape)+dis(44,46,shape))/(2*dis(42,45,shape))))/2)
def MAR(shape):
    return float((((dis(49,59,shape)+dis(50,58,shape)+dis(51,57,shape)+dis(52,56,shape)+dis(53,55,shape))/(2*dis(49,55,shape)))+((dis(61,67,shape)+dis(62,66,shape)+dis(63,65,shape))/(2*dis(61,65,shape))))/2)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Total_blinks=0 #Blinks
close_frame=0 # Number of frames for which eyes were closed ,used for calculating one blink
itime=time.time() #initial time
Avg_EAR=0 #The average EAR for the last few duration of time
blinks_per_minute=0 #Blinks per minute
Avg_blinks_per_minute=4.00 #Number of blinks per minute
t=0 #time passed in each cycle
n=1 #number of cycles
E=0 # Instantanious EAR
M=0 # Instantanious MAR
Earray=np.ones((1,20),dtype=float)*0.25 #Numpy Array for storing the last specified number of calculated EAR values
V=cv2.VideoCapture(0)
while(V.isOpened()):

    r,frame= V.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,1)
    #If no faces have been detected sets Avg values of EAR and MAR to defaults so as to not trigger the alarm
    if len(rects)==0:
        Avg_EAR=0.3
        E=0.3
        M=0.4
    for (i,rect) in enumerate(rects):#Each rect represents one face
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
    for i in range(19,0,-1):
        Earray[0,i]=Earray[0,i-1]

    Earray[0,0]=E
    Avg_EAR=np.sum(Earray)/20

    if Avg_EAR<0.22:
        cv2.putText(frame,"SLEEPY EYES",(60,450),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,255),11)
        winsound.Beep(350,200)
    if M>1.35:
        cv2.putText(frame,"YAWN",(200,250),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),15)
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
V.release()
cv2.destroyAllWindows()
