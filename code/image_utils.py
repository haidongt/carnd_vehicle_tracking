from scipy.misc import imresize
import cv2
import numpy as np
from train_svm import *
from scipy.ndimage.measurements import label


class VehicleDetector:
    def __init__(self,
        image_size=None,
        calibration_mtx=None,
        calibration_dist=None,
        perspective_src=None,
        perspective_dst=None,
        mask_vertices=None,
        clf=None,
        precalculated_windows=None):
        self.image_size = image_size
        self.calibration_mtx = calibration_mtx
        self.calibration_dist = calibration_dist
        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.mask_vertices = mask_vertices

        self.Minv = cv2.getPerspectiveTransform(perspective_dst, perspective_src)
        self.past_detected_lanes = []
        self.clf = clf
        self.frame_count = 0

        self.use_precalculated_windows = precalculated_windows is not None
        if precalculated_windows is None:
            self.use_precalculated_windows = False
            self.windows_by_frame = []
        else:
            self.use_precalculated_windows = True
            self.windows_by_frame = precalculated_windows

    def get_windows(self):
        windows = []
        windows.extend(self._get_windows(xy_window=(200, 200), xy_overlap=(0.75, 0.75),
            x_start_stop=[0, self.image_size[1]], y_start_stop=[int(self.image_size[0]/2), self.image_size[0]]))
        windows.extend(self._get_windows(xy_window=(128, 128), xy_overlap=(0.75, 0.75),
            x_start_stop=[0, self.image_size[1]], y_start_stop=[int(self.image_size[0]/2), int(self.image_size[0]*0.9)]))
        windows.extend(self._get_windows(xy_window=(64, 64), xy_overlap=(0.75, 0.75),
            x_start_stop=[int(self.image_size[1]*0.1), int(self.image_size[1]*0.9)], y_start_stop=[int(self.image_size[0]/2), int(self.image_size[0]*0.7)]))
        return windows

    def _get_windows(self, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = self.image_size[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = self.image_size[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def draw_windows(self, img, bboxes, color=(0, 0, 255), thick=3):
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def search_windows(self, img, windows):
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

            images = np.zeros((1, 64, 64, 3), dtype=np.uint8)
            images[0, :, :, :] = test_img
            prediction = self.clf.predict(images) 

            if prediction == 1:
                on_windows.append(window)
        return on_windows

    def get_heat_map(self):
        heatmap = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint32)
        for i in range(self.frame_count - 10, self.frame_count + 1):
            if i < 0:
                continue
            for box in self.windows_by_frame[i]:
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0], 0] += 1

        threshold = 20
        heatmap[heatmap <= threshold] = 0
        heatmap[heatmap > threshold] = 200

        return heatmap

    def get_bbox(self, heatmap):
        labels = label(heatmap)
        bboxes = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)
        return bboxes


    def process_frame(self, img):

        if self.use_precalculated_windows:
            on_windows = self.windows_by_frame[self.frame_count]
        else:
            windows = self.get_windows()
            on_windows = self.search_windows(img, windows)
            self.windows_by_frame.append(on_windows)
        
        heatmap = self.get_heat_map()
        bbox = self.get_bbox(heatmap)
        self.frame_count = self.frame_count + 1

        img = self.draw_windows(img, bbox)
        return img