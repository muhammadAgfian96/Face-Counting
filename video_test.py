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
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
mode_draw = False
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
thresh = 0.5
scales = [1024, 1980]
count = 1
gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
im_scale = 1.0
cctv='/media/HDD/LATIHAN_FIAN/insightFace/insightface/RetinaFace/ch04_20200229143636.mp4'

## ------------- RETINA END ----------------


cap = cv2.VideoCapture(2)
time.sleep(1)
pause_vid=False
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
		scales = [im_scale]
		flip = True

		for c in range(count):
			faces, landmarks = detector.detect(image, thresh, scales=scales, do_flip=flip)
            # print(c, faces)

		if faces is not None:
			print('find', faces.shape[0], 'faces')
			for i in range(faces.shape[0]):
				#print('score', faces[i][4])
				box = faces[i].astype(np.int)
				#color = (255,0,0)
				color = (0,0,255)
				cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
				if landmarks is not None:
					landmark5 = landmarks[i].astype(np.int)
					#print(landmark.shape)
					for l in range(landmark5.shape[0]):
						print(landmark5)
						color = (0,0,255)
						if l==0 or l==3:
							color = (0,255,0)
						cv2.circle(image, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

# -------------------- END MAIN ----------------------------

		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()
