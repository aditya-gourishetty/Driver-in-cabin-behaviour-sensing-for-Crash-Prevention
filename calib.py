import cv2
import dlib
import numpy as np
from imutils.face_utils import *
import matplotlib.pyplot as plt

# Function to change the dlib shape object to numpy array
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

def Calibration():
	e = [[]] * 2
	Avg_EAR = 0
	detector = dlib.get_frontal_face_detector()									# Face detector from dlib
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")	# Predictor for facial landmarks
	v = cv2.VideoCapture(0)														# Created a video capture object
	while v.isOpened():
		r, frame = v.read()														# Reading a frame
		frame = cv2.flip(frame, 1)
		clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# frame = clahe.apply(gray)
		rectangles = detector(frame)											# Finding rectangles of facial features
		for (i, rectangle) in enumerate(rectangles):							# For each rectangle
			# (x,y,w,h)=rect_to_bb(rectangle)
			shape = predictor(frame, rectangle)
			shape = shape_to_np(shape)
			for (x, y) in shape[36:48]:
				cv2.circle(frame, (x, y), 2, (0, 255, 0), - 1)
			for (x, y) in shape[48:68]:
				cv2.circle(frame, (x, y), 2, (255, 0, 0), - 1)
			E=EAR(shape)
			e[i].append(E)
			Avg_EAR = sum(e[i]) / len(e[i])
			cv2.putText(frame, str(i) + " Avg EAR={:1.2f}".format(Avg_EAR), (30, 30),cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
			cv2.putText(frame, "EAR={:1.2f}".format(E),(200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

		cv2.imshow('frame', frame)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
	
	# plt.plot(smooth(e[0], 20), 'r')
	# # plt.plot(smooth(e[0], 20), markevery = 100)
	# # plt.show()
	# plt.plot(smooth(e[1], 20), 'g')
	plt.show()
	# print(Avg_EAR, smooth(e[0], len(e[0])))
	cv2.destroyAllWindows()
	return (Avg_EAR + 0.02)
