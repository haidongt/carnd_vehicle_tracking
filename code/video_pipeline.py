import glob
import image_utils
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip


dist_pickle = pickle.load( open( "../camera_cal/camera_calibration_result.p", "rb" ) )
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

lane_detector = image_utils.LaneDetector(
	image_size=IMAGE_SIZE,
	calibration_mtx=mtx,
	calibration_dist=dist,
	perspective_src=SRC,
	perspective_dst=DST,
	mask_vertices=ROI_VERTICES)

VIDEOS = ["../videos/project_video.mp4", "../videos/challenge_video.mp4", "../videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 0

clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
project_clip = clip1.fl_image(lane_detector.process_frame)

project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
project_clip.write_videofile(project_output, audio=False)