
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
from centroidtracker import PointTracker, CentroidTracker
from intersect import find_intersection
import time

from collections import deque

oke_line = False
oke_roi = False
ct = CentroidTracker()
crop_img = np.array([0,0,0])
refPt = []
known_id, known_count_id = [], {}
manyPoints ={}
counting_all_id=deque(maxlen=32)
num_count = 0 
draw_line, select_roi = False, False 
singgung = False
newFace = deque(maxlen=5)
offset_x, offset_y = 0,0

def inputFace(src, face):
	x,y,z = face.shape
	img[0:x, 0:y] = face  # make it PIP
	return image
	


def make_line(event, x, y, flags, param):
	# grab references to the global 
	global refPt, draw_line, refRct, select_roi, oke_line, oke_roi
	global x_old, y_old, x_old_rct, y_old_rct, img2, draw, pause_vid
	if event == cv2.EVENT_LBUTTONDOWN:
		if draw_line == False:
			# step 1
			refPt = [(x,y)]	
			draw_line = True
			x_old = x
			y_old = y
			pause_vid = True
			print("LINE True")
		elif draw_line:
			# step 3
			refPt.append((x, y))
			print(refPt)
			draw_line = False
			cv2.line(draw, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.imshow("image", draw)
			pause_vid = False
			print("line FALSE")
			oke_line = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if draw_line:
			draw = img2.copy()
			cv2.line(draw, (x_old, y_old),(x,y), (0,0,255), thickness=2)
			cv2.imshow("image", draw)
		if select_roi:
			draw = img2.copy()
			cv2.rectangle(draw, (x_old_rct, y_old_rct),(x,y), (4, 200, 150), thickness=2)
			cv2.imshow("image", draw)
	# ---------- SELECT ROI ---------------
	elif event == cv2.EVENT_MBUTTONDOWN:
		if select_roi == False:
			# step 1
			refRct = [(x,y)]	
			select_roi = True
			x_old_rct = x
			y_old_rct = y
			pause_vid = True
			print("select_roi True")
		elif select_roi:
			# step 3
			refRct.append((x, y))
			print(refRct)
			select_roi = False
			cv2.rectangle(draw, refRct[0], refRct[1], (4, 200, 150), 2)
			cv2.imshow("image", draw)
			pause_vid = False
			print("select_roi FALSE")
			oke_roi=True
		if mode_draw:
			# step 2
			# print("masuk")
			draw = img2.copy()
			cv2.line(draw, (x_old, y_old),(x,y), (0,0,255), thickness=2)
			cv2.imshow("image", draw)

# -------------- START RETINA -----------------
thresh = 0.5
scales = [1024, 1980]
count = 1
gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
im_scale = 1.0
cctv='/media/HDD/LATIHAN_FIAN/insightFace/insightface/RetinaFace/ch04_20200229143636.mp4'
prays = '/media/HDD/age_gender_trial/DarkflowTensorRT3/demo.mp4'
## ------------- RETINA END ----------------


cap = cv2.VideoCapture(0)
time.sleep(1)
pause_vid=False
cv2.namedWindow("image")
cv2.setMouseCallback("image", make_line)
print("hasd")
# keep looping until the 'q' key is pressed
while True:
	grab, image = cap.read()
	if grab:
		aslina = image.copy()
		# image = cv2.resize(image, (960, 480), cv2.INTER_AREA)
		first = True
		while pause_vid:
			draw = image.copy()
			if first ==True:
				img2 = image.copy()
			cv2.setMouseCallback("image", make_line)
			cv2.waitKey(1)
			first = False
			known_count_id ={}
		if len(refPt)>=2 and len(refPt)%2 ==0:
			cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)
		if oke_roi:
			# print("masuk draw", refRct)
			cv2.rectangle(image, refRct[0], refRct[1], (4, 200, 150), 2)
			roi_process = image[refRct[0][1]:refRct[1][1], refRct[0][0]:refRct[1][0]]
			offset_x = refRct[0][0]
			offset_y = refRct[0][1]

#------------------- MAIN PROGRAM --------------------------
		scales = [im_scale]
		flip = True

		if oke_roi:
			faces, landmarks = detector.detect(roi_process, thresh, scales=scales, do_flip=flip)
		else:
			faces, landmarks = detector.detect(image, thresh, scales=scales, do_flip=flip)
		
		rects = []
		if faces is not None:
			print('find', faces.shape[0], 'faces')
			for i in range(faces.shape[0]):
				box = faces[i].astype(np.int)
				score = box[4]
				color = (0,0,255)
				# cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
				rect = np.asarray([box[0]+offset_x, box[1]+offset_y, 
								  box[2]+offset_x, box[3]+offset_y]).astype("int")
				rects.append(rect)

		objects = ct.update(rects)
		# for () in objects.items():
		# 	print("???",haha)
		for (objectID, paket) in objects.items():
			centroid, bbox = paket
			if objectID not in known_id:
				known_id.append(objectID)
				manyPoints[objectID] = PointTracker()
			manyPoints[objectID].update_ctr((centroid[0], centroid[1]))
			text = "ID {}".format(objectID)
			print(bbox)
			cv2.rectangle(image, (bbox[0], bbox[1]), 
								 (bbox[2], bbox[3]), 
						 		 (12,255,23), 2)
			cv2.putText(image, text, 
						(centroid[0] - 10, centroid[1] - 10), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
						(0, 255, 0), 2)
			cv2.circle(image, 
						(centroid[0], centroid[1]), 
						4, (0, 125, 0), -1)


			pointss = manyPoints[objectID].getPts()
			
			# drawing tracking
			# for i in range(1, len(pointss)):
			# 	if pointss[i - 1] is None or pointss[i] is None:
			# 		continue
			# 	thickness = int(np.sqrt(16 / float(i + 1)) * 2.5)
			# 	cv2.line(image, pointss[i - 1], pointss[i], 
			# 			(1+thickness*20, 255-thickness*20, 255), 
			# 			thickness)

			thickness=2
			cv2.line(image, pointss[0], pointss[-1], (1+thickness*20, 255-thickness*20, 255), thickness)

			# print(pointss, refPt)
			if len(refPt)>0:
				if len(pointss)>2:
					singgung = find_intersection((refPt[0][0], refPt[0][1]), 
													(refPt[1][0], refPt[1][1]), 
													pointss[-1], pointss[0])
					if singgung and objectID not in counting_all_id:
						known_count_id[objectID]=1
						counting_all_id.append(objectID)
						ofzide = 30
						crop_img = aslina[bbox[1]-ofzide:bbox[3]+ofzide, 
											bbox[0]-ofzide:bbox[2]+ofzide]
						if crop_img.shape[0]>0 and crop_img.shape[1]>0:
							crop_img = cv2.resize(crop_img, (60,60))
							cv2.putText(crop_img,f"ob: {objectID}", (3,50), 
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
										(0,255,5), 2)
							newFace.appendleft(crop_img)

		if len(newFace)>0:
			for i,face in enumerate(newFace, start=1):
				image[0:60, 60*(i-1):60*(i)] = face

		# counting_all_id=[]
		num_count = len(known_count_id)
		cv2.putText(image, "counting:"+str(num_count), 
							(offset_x+10, offset_y+20), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
							(10,10,225), 2)
# -------------------- END MAIN ----------------------------
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()