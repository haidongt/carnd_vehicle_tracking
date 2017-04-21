import glob
import image_utils
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from train_svm import *


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

with open('../svm_model.p', 'rb') as f:
        model = pickle.load(f)

windows_by_frame = pickle.load( open( "../windows_by_frame.p", "rb" ) )
#windows_by_frame = None

vehicle_detector = image_utils.VehicleDetector(
    image_size=IMAGE_SIZE,
    calibration_mtx=mtx,
    calibration_dist=dist,
    perspective_src=SRC,
    perspective_dst=DST,
    mask_vertices=ROI_VERTICES,
    model=model,
    precalculated_windows=windows_by_frame)

VIDEOS = ["../test_video.mp4", "../project_video.mp4", "../videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 1

clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
project_clip = clip1.fl_image(vehicle_detector.process_frame)

project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
project_clip.write_videofile(project_output, audio=False)
if windows_by_frame is None:
    with open('../windows_by_frame.p', 'wb') as f:
            pickle.dump(vehicle_detector.windows_by_frame, f)