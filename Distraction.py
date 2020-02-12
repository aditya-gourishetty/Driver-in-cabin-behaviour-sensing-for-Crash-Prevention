import dlib
import cv2
from imutils import face_utils
import numpy as np
import math
import winsound
c = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\Python27\Lib\site-packages\shape_predictor_68_face_landmarks1.dat")
v = cv2.VideoCapture(0)


while v.isOpened():
    r, frame = v.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (b, l) = gray.shape

    rects = detector(gray, 1)

    if not detector(gray):
        c = c+1
        print "nif"

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)
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
        #cv2.putText(frame, "q".format(q), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1)

        rotation_matrix, j = cv2.Rodrigues(ro)
        q = math.sqrt(pow(ro[1], 2) + pow(ro[2], 2))
        cv2.putText(frame, "q={}".format(q), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        print "q={}".format(q)
        #print "rotation matrix={}".format(rotation_matrix)

        if q >= 0.45:
            c = c+1
        else:
            c = 0
    if c >= 6:
        cv2.putText(frame, 'WARNING : DISTRACTED DRIVER', (100, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        #winsound.beep(700, 200)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break



