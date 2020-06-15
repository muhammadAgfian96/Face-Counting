import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
from utils import draw_boxes

from deep_sort import build_tracker
cfg ={
    "REID_CKPT": "./deep_sort/deep/checkpoint/ckpt.t7",
    "MAX_DIST": 0.2,
    "MIN_CONFIDENCE": 0.3,
    "NMS_MAX_OVERLAP": 0.5,
    "MAX_IOU_DISTANCE": 0.7,
    "MAX_AGE": 70,
    "N_INIT": 3,
    "NN_BUDGET": 100,
}

deepsort = build_tracker(cfg, use_cuda=True)


thresh = 0.5
count = 1

gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
im_scale = 1.0


cctv = '/media/HDD/LATIHAN_FIAN/insightFace/insightface/RetinaFace/ch04_20200229143636.mp4'

cap = cv2.VideoCapture(cctv)

while True:
    grab, img = cap.read()

    if grab:
        img = cv2.resize(img, (960,480), cv2.INTER_AREA)
        print(img.shape)
        scales = [im_scale]
        flip = True

        # do detection
        faces_xyxy, landmarks = detector.detect(img, thresh, scales=scales, do_flip=True)
        print(faces_xyxy, landmarks.shape)

        if faces_xyxy is not None:
            print('find', faces_xyxy.shape[0], 'faces')
            faces_xywh = []
            score_fit = []
            for i in range(faces_xyxy.shape[0]):
                box = faces_xyxy[i].astype(np.int)
                bbox_xywh = [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]
                faces_xywh.append(bbox_xywh)
                score_fit.append(faces_xyxy[i][4])
            outputs = deepsort.update(np.array(faces_xywh), np.array(score_fit), img)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,-1]
                ori_im = draw_boxes(img, bbox_xyxy, identities)
        
        
        cv2.imshow('test_detector', img)
        if cv2.waitKey(1) & 0xFF== ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#   filename = './detector_test.jpg'
#   print('writing', filename)
#   cv2.imwrite(filename, img)

