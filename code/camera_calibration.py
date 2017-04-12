import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imresize
import glob
import pickle
from scipy import signal

IMAGE_SIZE = (720, 1280, 3)
nx = 9
ny = 6

images = glob.glob("../camera_cal/calibration*.jpg")

objpoints = []
imgpoints = []

objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

imgs = []

for fname in images:
    img = cv2.imread(fname)
    if img.shape[0] != IMAGE_SIZE[0] or img.shape[1] != IMAGE_SIZE[1]:
    	img = imresize(img, IMAGE_SIZE)
    imgs.append(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        
img_size = (imgs[0].shape[1], imgs[0].shape[0])

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save calibration matrix and distortion coefficients
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("../camera_cal/camera_calibration_result.p", "wb"))