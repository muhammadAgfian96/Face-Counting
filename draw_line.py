# import the necessary packages
import argparse
import cv2
import time
import numpy as np
import sys
import datetime
import os
import glob
from retinaface import RetinaFace

refPt = []
mode_draw = False
pause_vid=False

def make_line(event, x, y, flags, param):
	# grab references to the global 
	global refPt, mode_draw, x_old, y_old, img2, draw, pause_vid
	if event == cv2.EVENT_LBUTTONDOWN:
		if mode_draw == False:
			# step 1
			refPt = [(x,y)]	
			mode_draw = True
			x_old = x
			y_old = y
			pause_vid = True
			print("CROPPING True")
		elif mode_draw:
			# step 3
			refPt.append((x, y))
			print(refPt)
			mode_draw = False
			cv2.line(draw, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.imshow("image", draw)
			pause_vid = False
			print("CROPPING FALSE")
	elif event == cv2.EVENT_MOUSEMOVE:
		if mode_draw:
			# step 2
			# print("masuk")
			draw = img2.copy()
			cv2.line(draw, (x_old, y_old),(x,y), (0,0,255), thickness=2)
			cv2.imshow("image", draw)


# load the image, clone it, and setup the mouse callback function


# -------------- START RETINA -----------------

## ------------- RETINA END ----------------


cap = cv2.VideoCapture(cctv)
time.sleep(1)
cv2.namedWindow("image")
cv2.setMouseCallback("image", make_line)
print("hasd")
# keep looping until the 'q' key is pressed
while True:
	grab, image = cap.read()
	if grab:
		image = cv2.resize(image, (720, 480), cv2.INTER_AREA)
		first = True
		while pause_vid:
			draw = image.copy()
			if first ==True:
				img2 = image.copy()
			cv2.setMouseCallback("image", make_line)
			cv2.waitKey(1)
			first = False
		if len(refPt)>=2 and len(refPt)%2 ==0:
			cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)

#------------------- MAIN PROGRAM --------------------------

# -------------------- END MAIN ----------------------------

		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()
