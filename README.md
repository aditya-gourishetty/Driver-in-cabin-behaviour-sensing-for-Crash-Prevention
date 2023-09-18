# Driver-in-cabin-behaviour-sensing-for-Crash-Prevention
Driver in-cabin behaviour sensing for Crash Prevention

## MOTIVATION
The Global status report on road safety 2018, launched by WHO in December 2018, highlights that the number of annual road traffic deaths has reached 1.35 million. Driver fatigue and drowsiness are significant factors in a large number of vehicle accidents. 
Drowsy driving usually happens when a driver has not slept enough, but it can also happen due to untreated sleep disorders, medications, drinking alcohol, or shift work.
Drowsiness—
- Makes drivers less able to pay attention to the road.
- Slows reaction time if you have to brake or steer suddenly.
- Affects a driver’s ability to make good decisions

## OBJECTIVE
Driver drowsiness detection system is a car safety technology which aims to prevent accidents caused by the driver getting drowsy. 
This project aims to use the behavioral symptoms given off by the driver to detect drowsiness and subsequently set off an alarm or a vibrator attached to the seatbelt.

## FACIAL LANDMARKS
The key points of interest on the shape i.e. the face which help in defining the facial features such as ,nose ,eyes ,jaw ,mouth etc. are called facial landmarks .We use python's dlib library which contains machine learning algorithms, including computer vision for this purpose.
Steps for detecting facial landmarks -
![image](https://github.com/aditya-gourishetty/Driver-in-cabin-behaviour-sensing-for-Crash-Prevention/assets/145345383/2466e84e-6b34-4d8d-a742-c32c6618723b)

The end result is we get the location of 68 (x, y)-coordinates that map to facial structures on the face
![image](https://github.com/aditya-gourishetty/Driver-in-cabin-behaviour-sensing-for-Crash-Prevention/assets/145345383/93981d3f-7859-4b5c-806c-0821f70ac81d)

## EYE ASPECT RATIO
From the eye landmarks detected in the image, we derive the EAR which is used  to estimate the state of the eye.
Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye , and then working clockwise around the remainder of the region.
p1,...,p6 are the 2D landmark locations, depicted in Fig. 1.
The EAR is mostly constant when an eye is open and will fall to zero when the eye is closed. It is partially person and head pose insensitive.

![image](https://github.com/aditya-gourishetty/Driver-in-cabin-behaviour-sensing-for-Crash-Prevention/assets/145345383/a2fbf745-cf74-4ecf-b8e6-a351351b7428)

Eye landmarks of a closed eye.
Eye aspect ratio plotted over time. A single blink indicated by the dip is present.
