import glob
import image_utils
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from train_svm import *


fnames = glob.glob("../test_images/*.jpg")

dist_pickle = pickle.load( open( "../camera_calibration_result.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

IMAGE_SIZE = (720, 1280, 3)
ROI_VERTICES = np.array([[(0,720),(550, 470), (800, 470), (1280, 720)]], dtype=np.int32)


OFFSET = 250
SRC = np.float32([
    (132, 703),
    (540, 466),
    (740, 466),
    (1147, 703)])

DST = np.float32([
    (SRC[0][0] + OFFSET, 720),
    (SRC[0][0] + OFFSET, 0),
    (SRC[-1][0] - OFFSET, 0),
    (SRC[-1][0] - OFFSET, 720)])

with open('../svm_final.p', 'rb') as f:
        clf = pickle.load(f)

lane_detector = image_utils.VehicleDetector(
	image_size=IMAGE_SIZE,
	calibration_mtx=mtx,
	calibration_dist=dist,
	perspective_src=SRC,
	perspective_dst=DST,
	mask_vertices=ROI_VERTICES,
	clf=clf)

for fname in fnames:
	img = cv2.imread(fname)
	output_img = lane_detector.process_frame(img)
	output_fname = '../test_images/output/' + fname[15:-4] + '_output.jpg'
	cv2.imwrite(output_fname,output_img)
	
