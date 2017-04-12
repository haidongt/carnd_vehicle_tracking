import pickle
import cv2
from scipy.misc import imresize, imread
import glob

dist_pickle = pickle.load( open( "../camera_cal/camera_calibration_result.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

IMAGE_SIZE = (720, 1280, 3)

fnames = glob.glob("../camera_cal/calibration*.jpg")

for fname in fnames:
	img = cv2.imread(fname)
	img = imresize(img, IMAGE_SIZE)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	output_fname = fname[:-4] + '_output.jpg'
	cv2.imwrite(output_fname,dst)




