from scipy.misc import imresize
import cv2
import numpy as np

class Lane():
    def __init__(self, max_y):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

        self.line_fit = None

        self.max_y = max_y

        self.curvature = None

    def calculate(self):
        self.line_base_pos = self.line_fit[0]*self.max_y**2 + self.line_fit[1]*self.max_y + self.line_fit[2]
        self.calculate_curvature()

    def calculate_curvature(self):
        x = self.allx
        y = self.ally
        y_eval = np.max(y)
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/720 # 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])



class LaneDetector:
    def __init__(self,
        image_size=None,
        calibration_mtx=None,
        calibration_dist=None,
        perspective_src=None,
        perspective_dst=None,
        mask_vertices=None):
        self.image_size = image_size
        self.calibration_mtx = calibration_mtx
        self.calibration_dist = calibration_dist
        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.mask_vertices = mask_vertices

        self.Minv = cv2.getPerspectiveTransform(perspective_dst, perspective_src)
        self.past_detected_lanes = []

    def undistort(self, img):
        img = imresize(img, self.image_size)
        dst = cv2.undistort(img, self.calibration_mtx, self.calibration_dist,
            None, self.calibration_mtx)
        return dst

    def perspective_transform(self, img):
        M = cv2.getPerspectiveTransform(self.perspective_src, self.perspective_dst)
        warped = cv2.warpPerspective(img, M, (self.image_size[1], self.image_size[0]), flags=cv2.INTER_LINEAR)
        return warped

    def generate_thresholded_image(self, img, v_cutoff=0):
        """
        Generates a binary mask selecting the lane lines of an street scene image.
        :param img: RGB color image
        :param v_cutoff: vertical cutoff to limit the search area
        :return: binary mask
        """
        window = img[v_cutoff:, :, :]
        yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
        yuv = 255 - yuv
        hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
        chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
        gray = np.mean(chs, 2)

        s_x = abs_sobel(gray, orient='x', sobel_kernel=3)
        s_y = abs_sobel(gray, orient='y', sobel_kernel=3)

        grad_dir = gradient_direction(s_x, s_y)
        grad_mag = gradient_magnitude(s_x, s_y)

        ylw = extract_yellow(window)
        highlights = extract_highlights(window[:, :, 0])

        mask = np.zeros(img.shape[:-1], dtype=np.uint8)

        
        mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &
                            (s_y >= 25) & (s_y <= 255)) |
                           ((grad_mag >= 30) & (grad_mag <= 512) &
                            (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                           (ylw == 255) |
                           (highlights == 255)] = 1
                           


        #mask[v_cutoff:, :][(hls[:, :, 2] > 100)] = 200

        mask = binary_noise_reduction(mask, 4)
        return mask

    def apply_region_of_interest(self, img):
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, self.mask_vertices, ignore_mask_color)

        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def find_lanes(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 


        # lanes
        left_lane = Lane(self.image_size[0])
        right_lane = Lane(self.image_size[0])
        left_lane.allx = leftx
        left_lane.ally = lefty
        right_lane.allx = rightx
        right_lane.ally = righty
        self.past_detected_lanes.insert(0, (left_lane, right_lane))
        if len(self.past_detected_lanes) > 5:
            self.past_detected_lanes.pop()


        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_lane.line_fit = left_fit
        right_lane.line_fit = right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return out_img

    def unwarp(self, undist, warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.image_size[1], self.image_size[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    def fusion_debug_info(self, original, debug_img):
        # draw perspective view at upper right
        debug_img = imresize(debug_img, (int(self.image_size[0]/2), int(self.image_size[1]/2), 3))
        original[0:int(self.image_size[0]/2), int(self.image_size[1]/2):] = debug_img

        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(original, 'left: y = %0.4fx^2 + %0.2fx + %0.2f' % (self.past_detected_lanes[0][0][0], self.past_detected_lanes[0][0][1], self.past_detected_lanes[0][0][2]), (50, 50), font, 1, (255, 255, 255), 2)
        #cv2.putText(original, 'right: y = %0.4fx^2 + %0.2fx + %0.2f' % (self.past_detected_lanes[0][1][0], self.past_detected_lanes[0][1][1], self.past_detected_lanes[0][1][2]), (50, 80), font, 1, (255, 255, 255), 2)
        
        self.past_detected_lanes[0][0].calculate()
        self.past_detected_lanes[0][1].calculate()

        curvature_left = 0
        curvature_right = 0

        for past_detected_lane in self.past_detected_lanes:
            curvature_left = curvature_left + past_detected_lane[0].curvature
            curvature_right = curvature_left + past_detected_lane[1].curvature
        curvature_left = curvature_left / len(self.past_detected_lanes)
        curvature_right = curvature_right / len(self.past_detected_lanes)


        cv2.putText(original, 'left curvature(m): %s' % curvature_left, (50, 110), font, 1, (255, 255, 255), 2)
        cv2.putText(original, 'right curvature(m): %s' % curvature_right, (50, 140), font, 1, (255, 255, 255), 2)
        #cv2.putText(original, 'left: %s' % self.past_detected_lanes[0][0].line_base_pos, (50, 170), font, 1, (255, 255, 255), 2)
        #cv2.putText(original, 'right: %s' % self.past_detected_lanes[0][1].line_base_pos, (50, 200), font, 1, (255, 255, 255), 2)
        cv2.putText(original, 'position(m): %s' % (((self.past_detected_lanes[0][0].line_base_pos + self.past_detected_lanes[0][1].line_base_pos)/2-680)*3.7/720), (50, 170), font, 1, (255, 255, 255), 2)

        '''y_eval = self.image_size[0] - 1


        left_curverad = ((1 + (2*self.past_detected_lanes[0][0][0]*y_eval + self.past_detected_lanes[0][0][1])**2)**1.5) / np.absolute(2*self.past_detected_lanes[0][0][0])
        right_curverad = ((1 + (2*self.past_detected_lanes[0][1][0]*y_eval + self.past_detected_lanes[0][1][1])**2)**1.5) / np.absolute(2*self.past_detected_lanes[0][1][0])
        cv2.putText(original, 'left: %s' % left_curverad, (50, 110), font, 1, (255, 255, 255), 2)
        cv2.putText(original, 'right: %s' % right_curverad, (50, 140), font, 1, (255, 255, 255), 2)
        
        #self.past_detected_lanes
        '''
    
    def process_frame(self, img):
        if self.calibration_mtx is not None:
            undist = self.undistort(img)
        img = self.generate_thresholded_image(undist)
        img = self.apply_region_of_interest(img)
        img = self.perspective_transform(img)
        debug_img = self.find_lanes(img)
        img = self.unwarp(undist, img)

        # Debug fusion
        self.fusion_debug_info(img, debug_img)
        return img


def abs_sobel(img_ch, orient='x', sobel_kernel=3):
    """
    Applies the sobel operation on a gray scale image.

    :param img_ch:
    :param orient: 'x' or 'y'
    :param sobel_kernel: an uneven integer
    :return:
    """
    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)
    else:
        raise ValueError('orient has to be "x" or "y" not "%s"' % orient)

    sobel = cv2.Sobel(img_ch, -1, *axis, ksize=sobel_kernel)
    abs_s = np.absolute(sobel)

    return abs_s


def gradient_magnitude(sobel_x, sobel_y):
    """
    Calculates the magnitude of the gradient.
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return abs_grad_mag.astype(np.uint16)


def gradient_direction(sobel_x, sobel_y):
    """
    Calculates the direction of the gradient. NaN values cause by zero division will be replaced
    by the maximum value (np.pi / 2).
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)


def extract_yellow(img):
    """
    Generates an image mask selecting yellow pixels.
    :param img: image with pixels in range 0-255
    :return: Yellow 255 not yellow 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))

    return mask

def extract_highlights(img, p=99.9):
    """
    Generates an image mask selecting highlights.
    :param p: percentile for highlight selection. default=99.9
    :param img: image with pixels in range 0-255
    :return: Highlight 255 not highlight 0
    """
    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask

def binary_noise_reduction(img, thresh):
    """
    Reduces noise of a binary image by applying a filter which counts neighbours with a value
    and only keeping those which are above the threshold.
    :param img: binary image (0 or 1)
    :param thresh: min number of neighbours with value
    :return:
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img