import cv2
import numpy as np

from collections import deque

from features import roi_image


class Line:
    """ Gather and update lane lines information. """

    # Convertion of image to real world dimensions
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self, cache=20):
        # Memory size
        self.cache = cache
        # Current and recent line detection status
        self.detected = False
        self.recent_detections = deque(maxlen=cache * 2)
        # Current active lane line pixel coordinates
        self.x = None
        self.y = None
        # current and recent coefficients of the fitted lane line polynomial
        self.coeff = None
        self.recent_coeff = deque(maxlen=cache)
        # Current radius of line curvature in meters
        self.radius_of_curvature = None
        # Current line offset from vehicle location
        self.offset_from_vehicle = None

    def update(self, x, y, warped_binary_frame):
        """ Update the lane line with the new determined pixel coordinates.

        - Sanity check of new polynomial fit (coefficient deviations too high?)
        - Depending on the sanity check, setting to lost or detected status
        - Update the current radius of curvature
        - Update the current offset of the line from the vehicle location

        Args:
             x: x-coordinates of the active pixels in the warped frame
             y: y-coordinates of the active pixels in the warped frame
             warped_binary_frame: the current warped binary frame
        """

        if x.any() and y.any():
            # If pixels were found check if the the new polynomial coefficients
            # do not differ to much from the recent ones.
            # If the deviations are to high, set the line to lost status
            if len(self.recent_detections) >= self.cache:
                # Start the check if the line is not too long in lost status
                if np.sum(np.array(self.recent_detections)[-self.cache:]) != 0:
                    self.coeff = np.polyfit(y, x, deg=2)
                    deviation = np.abs(1 - self.coeff / self.avg_coeff)
                    if deviation[2] > 0.3 or deviation[0] > 40:
                        # If deviation is to high, set to lost status
                        self.set_lost_status()
                        return
                else:
                    # Reset the memory if the line is lost for too long
                    self.recent_coeff = deque(maxlen=self.cache)

            # If still here update the line with the current data
            self.detected = True
            self.recent_detections.append(self.detected)
            # Update the current lane line pixel coordinates
            self.x = x
            self.y = y
            # Update the current coeff by the new lane line pixels
            self.coeff = np.polyfit(y, x, deg=2)
            # Append current polynomial coeffs to the queue of recents
            self.recent_coeff.append(self.coeff)
            # Update the offset of the lane line from the vehicle loaction
            self.update_offset_from_vehicle(warped_binary_frame)
            # Update curvature radius at the vehicle location
            self.update_radius_of_curvature_m(warped_binary_frame)

        else:
            # If no pixels were found set lost status
            self.set_lost_status()

    def set_lost_status(self):
        """ Set the line to lost status.

        In lost status the recent coefficients will be used to keep estimating
        the lane line.
        """
        self.detected = False
        self.recent_detections.append(self.detected)
        self.x = None
        self.y = None
        self.coeff = self.recent_coeff[-1]

    @property
    def avg_coeff(self):
        """ Computes the average coefficients of the last N=cache fits. """
        return np.mean(self.recent_coeff, axis=0)

    def evaluate_average_polynomial(self, y):
        """ Evalualte the average polynomial at the given y-coordinates.

        The evalution is perfomed with the average coefficients of the last
        N=cache fits.

        Args:
            y: the y-coordinates to evaluate the polynomial for
        Returns:
            the corresponding x-coordinates on the polynomial curve
        """
        A, B, C = self.avg_coeff
        x = A * y ** 2 + B * y + C
        return x

    def update_offset_from_vehicle(self, warped_frame_binary):
        """
        Update the line offset from the vehicle based on the last N=cache fits.

        Args:
            warped_frame_binary: the current warped binary frame
        """

        if self.detected:
            image_height, image_width = warped_frame_binary.shape
            # The vehicle location is estimated to be at the image bottom center
            x_vehicle = image_width // 2
            y_vehicle = image_height - 1
            # Evaluate the lane line polynomial on the vehicle y-location
            x_line_at_y_vehicle = self.evaluate_average_polynomial(y_vehicle)
            # Compute the offset and convert to meters
            offset_pixels = x_line_at_y_vehicle - x_vehicle
            offset_meters = offset_pixels * Line.xm_per_pix
        else:
            # If line is in lost status keep the previous offset
            offset_meters = self.offset_from_vehicle
        # Update the line offset
        self.offset_from_vehicle = offset_meters

    def update_radius_of_curvature_m(self, warped_frame_binary):
        """ Update the curvature radius [m] based on the last N=cache fits.

        Args:
            warped_frame_binary: the current warped binary frame
        """

        # The vehicle location is estimated to be at the bottom of the image
        y_vehicle = warped_frame_binary.shape[0] - 1
        # Coefficients of the recent polynomials in pixels
        A, B, C = self.avg_coeff
        # Convert coefficients for the use in meters
        A_m = A * self.xm_per_pix / self.ym_per_pix ** 2
        B_m = B * self.xm_per_pix / self.ym_per_pix
        # Compute the radius of curvature for the lane line
        curv = (1 + (2 * A_m * y_vehicle + B_m) ** 2) ** (3 / 2) / abs(2 * A_m)
        # Update the radius of curvature
        self.radius_of_curvature = curv

    def draw_polynomial(self, warped_binary_frame):
        """
        Draw the average lane line polynomial onto the warped binary frame.

        Args:
             warped_binary_frame: the current warped binary frame
        """

        # Generate x and y values for plotting
        image_height = warped_binary_frame.shape[0]
        y = np.linspace(0, image_height - 1, image_height)
        x = self.evaluate_average_polynomial(y)

        # Plot the lane line polynomial on the warped binary frame
        points = np.column_stack((x, y)).reshape(-1, 1, 2).astype(np.int32)

        return cv2.polylines(warped_binary_frame, [points],
                             isClosed=False, color=(255, 255, 0), thickness=10)


def warp(img, roi_src, roi_dst):
    """ Perform a perspective transform on the given undistorted image.

    Args:
        img: undistorted image of current frame, e.g. the binary feature mask
        roi_src: perspective transform source as ratios of image dims x and y
        roi_dst: perspective transform destination in the same format as src

    Returns:
        (warped_img, trsf_mtx, trsf_mtx_inv)
    """

    # Image dimensions
    x, y = img.shape[1], img.shape[0]

    src = np.float32([[(roi_src["x1"] * x, roi_src["y1"] * y),
                       (roi_src["x2"] * x, roi_src["y2"] * y),
                       (roi_src["x3"] * x, roi_src["y3"] * y),
                       (roi_src["x4"] * x, roi_src["y4"] * y)]])
    dst = np.float32([[(roi_dst["x1"] * x, roi_dst["y1"] * y),
                       (roi_dst["x2"] * x, roi_dst["y2"] * y),
                       (roi_dst["x3"] * x, roi_dst["y3"] * y),
                       (roi_dst["x4"] * x, roi_dst["y4"] * y)]])

    # Transformation matrix for the perspective transform and its inverse
    trsf_mtx = cv2.getPerspectiveTransform(src, dst)
    trsf_mtx_inv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image
    warped_img = cv2.warpPerspective(img, trsf_mtx, (x, y))

    return warped_img, trsf_mtx, trsf_mtx_inv


def draw_lane_on_unwarped_frame(frame, left_line, right_line, trsf_mtx_inv):
    """ Drawing of the unwarped lane lines and lane area to the current frame.

    Args:
        left_line: left Line instance
        right_line: right Line instance
        trsf_mtx_inv: inverse of the perspective transformation matrix
    """

    # Frame dimensions
    height, width = frame.shape[0:2]

    # Generate x and y values for plotting
    y = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
    left_x = left_line.evaluate_average_polynomial(y)
    right_x = right_line.evaluate_average_polynomial(y)

    # Create a green lane area between the left and right lane lines
    warped_lane_area = np.zeros_like(frame)  # Warped at first
    left_points = np.column_stack((left_x, y)).reshape((1, -1, 2)).astype(int)
    right_points = np.flipud(
        np.column_stack((right_x, y))).reshape((1, -1, 2)).astype(int)
    vertices = np.hstack((left_points, right_points))
    cv2.fillPoly(warped_lane_area, [vertices], (0, 255, 0))
    # Unwarp the lane area
    unwarped_lane = cv2.warpPerspective(
        warped_lane_area, trsf_mtx_inv, (width, height))
    # Overlay the unwarped lane area onto the frame
    green_lane_on_frame = cv2.addWeighted(frame, 1., unwarped_lane, 0.3, 0)

    # Draw the left and right lane polynomials into an empty and warped image
    warped_lanes = np.zeros_like(frame)
    left_points = np.column_stack((left_x, y)).reshape(-1, 1, 2)
    right_points = np.column_stack((right_x, y)).reshape(-1, 1, 2)
    warped_lanes = cv2.polylines(warped_lanes,
                                 [left_points.astype(np.int32)],
                                 isClosed=False, color=(255, 0, 0),
                                 thickness=30)
    warped_lanes = cv2.polylines(warped_lanes,
                                 [right_points.astype(np.int32)],
                                 isClosed=False, color=(0, 0, 255),
                                 thickness=30)
    # Unwarp the lane lines plot
    lane_lines = cv2.warpPerspective(
        warped_lanes, trsf_mtx_inv, (width, height))

    # Create a mask of the unwarped lane lines to shadow the frame background
    # a bit
    gray = cv2.cvtColor(lane_lines, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    # Black-out the area of the lane lines in the frame
    frame_bg = cv2.bitwise_and(
        green_lane_on_frame, green_lane_on_frame, mask=mask)
    # Combine with complete frame to shadow the area of the lane lines a bit
    shadowed_frame = cv2.addWeighted(frame_bg, 0.6, green_lane_on_frame, 0.4, 0)
    return cv2.addWeighted(shadowed_frame, 1.0, lane_lines, 1.0, 0)


def fit_lines_by_windows(warped_feature_binary,
                         left_line: Line, right_line: Line,
                         nwindows=9, margin=100, minpix=100):
    """ Detection of the lane lines by the sliding windows approach.

    The starting centers of the first windows of each lane line are determined
    by finding the peaks in a histogram computed for the bottom half of the
    image. In each window the active pixels are counted. If they surpass the
    minpix value the window will be recentered. After iterating over all windows
    up to the top of the image, a second order polynomial is fitted through
    the found pixels for each lane line. Both lane Line instances are updated
    and returned together with an information image of the found windows and
    polynomials.

    Args:
        warped_feature_binary: warped feature binary of the current frame
        left_line: the Line instance of the left lane line
        right_line: the Line instance of the right lane line
        nwindows: number of sliding windows per line distributed over the
         height of the image
        margin: left and right extent of each window in number of pixels
        minpix: number of pixels at which the window will be recentered
    Returns:
        (left_line, right_line, window_image)
    """

    # Histogram of the bottom half of the image
    image_height = warped_feature_binary.shape[0]
    hist = np.sum(warped_feature_binary[image_height // 2:, :], axis=0)

    # Determine the starting point for the left and right lane lines
    midpoint = np.int(hist.shape[0] // 2)
    left_hist = hist[:midpoint]
    if left_hist.sum() != 0:
        # Find the peak of the left halve of the histogram
        starting_left_window_center_x = np.argmax(left_hist)
    elif left_line.x is not None:
        # If no peak is found take the mean of the recent x-pixels
        starting_left_window_center_x = left_line.x.mean()
    else:
        # If there are no recent x-pixels start at the first quarter
        starting_left_window_center_x = midpoint // 2

    # Same for the right lane line
    right_hist = hist[midpoint:]
    if right_hist.sum() != 0:
        starting_right_window_center_x = np.argmax(right_hist) + midpoint
    elif right_line.x is not None:
        starting_right_window_center_x = right_line.x.mean()
    else:
        starting_right_window_center_x = midpoint // 2 + midpoint

    # Window height
    window_height = np.int(image_height // nwindows)

    # Activated frame pixel coordinates
    active_frame_pixel_coordinates = warped_feature_binary.nonzero()
    active_frame_pixels_x = np.array(active_frame_pixel_coordinates[1])
    active_frame_pixels_y = np.array(active_frame_pixel_coordinates[0])

    # Current position to be updated later for each window in nwindows
    current_left_window_center_x = starting_left_window_center_x
    current_right_window_center_x = starting_right_window_center_x

    # Create empty lists to receive left and right lane pixel indices
    all_left_lane_indices = []
    all_right_lane_indices = []

    # Create an output image to draw on and visualize the colored result
    window_img = np.dstack((
        warped_feature_binary,
        warped_feature_binary,
        warped_feature_binary))

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        window_bot_y = image_height - (window + 1) * window_height
        window_top_y = image_height - window * window_height

        # Find the four boundaries of the window
        left_window_left_x = int(current_left_window_center_x - margin)
        left_window_right_x = int(current_left_window_center_x + margin)
        right_window_left_x = int(current_right_window_center_x - margin)
        right_window_right_x = int(current_right_window_center_x + margin)

        # Draw the windows on the visualization image
        cv2.rectangle(window_img,
                      (left_window_left_x, window_bot_y),
                      (left_window_right_x, window_top_y),
                      (0, 255, 0), 2)
        cv2.rectangle(window_img,
                      (right_window_left_x, window_bot_y),
                      (right_window_right_x, window_top_y),
                      (0, 255, 0), 2)

        # Identify the indices of the active pixels within the window
        left_lane_indices = (
                (active_frame_pixels_x >= left_window_left_x)
                & (active_frame_pixels_x <= left_window_right_x)
                & (active_frame_pixels_y >= window_bot_y)
                & (active_frame_pixels_y <= window_top_y)).nonzero()[0]
        right_lane_indices = (
                (active_frame_pixels_x >= right_window_left_x)
                & (active_frame_pixels_x <= right_window_right_x)
                & (active_frame_pixels_y >= window_bot_y)
                & (active_frame_pixels_y <= window_top_y)).nonzero()[0]

        # Append these indices to the lists
        all_left_lane_indices.append(left_lane_indices)
        all_right_lane_indices.append(right_lane_indices)

        # If found pixels > minpix pixels, recenter next window
        if len(left_lane_indices) > minpix:
            current_left_window_center_x = np.int(
                np.mean(active_frame_pixels_x[left_lane_indices]))
        if len(right_lane_indices) > minpix:
            current_right_window_center_x = np.int(
                np.mean(active_frame_pixels_x[right_lane_indices]))

    # Concatenate the arrays of indices (up to now list of lists)
    all_left_lane_indices = np.concatenate(all_left_lane_indices)
    all_right_lane_indices = np.concatenate(all_right_lane_indices)

    # Update and fit the lane lines by the new pixel coordinates
    # image bottom is estimated vehicle location
    left_line.update(x=active_frame_pixels_x[all_left_lane_indices],
                     y=active_frame_pixels_y[all_left_lane_indices],
                     warped_binary_frame=warped_feature_binary)
    right_line.update(x=active_frame_pixels_x[all_right_lane_indices],
                      y=active_frame_pixels_y[all_right_lane_indices],
                      warped_binary_frame=warped_feature_binary)

    # Color the left and right lane regions with the detected active pixels
    if left_line.x is not None and left_line.y is not None:
        window_img[left_line.y, left_line.x] = [255, 0, 0]
    if right_line.x is not None and right_line.y is not None:
        window_img[right_line.y, right_line.x] = [0, 0, 255]

    # Draw the average polynomial lane lines onto the warped image
    window_img = left_line.draw_polynomial(window_img)
    window_img = right_line.draw_polynomial(window_img)

    return left_line, right_line, window_img


def fit_lines_by_recent_lines(warped_feature_binary,
                              left_line: Line, right_line: Line, margin=100):
    """
    Detection of the lane lines by using information of recently found lines.

    This approach uses the polynomials of previously found lines to determine
    the area in which to search for the current lane lines. The search area
    boundaries are determined by shifting the previous polynomials to the left
    and right by the number of pixels given by margin. New polynomials are then
    fitted though the active pixels found inside the left and right lane line
    search windows. Both lane Line instances are updated and returned together
    with an information image of the search areas and polynomials.

    Args:
        warped_feature_binary: warped feature binary of the current frame
        left_line: the Line instance of the left lane line
        right_line: the Line instance of the right lane line
        margin: left and right extent of each lane line polynomial window in
         number of pixels
    Returns:
        (left_line, right_line, window_image)
    """

    # Prepare image arrays to show the selection window and colored lane lines
    colored_lane_lines = np.dstack((warped_feature_binary,
                                    warped_feature_binary,
                                    warped_feature_binary)) * 255
    window_img = np.zeros_like(colored_lane_lines)

    # Activated pixel coordinates in the warped frame
    active_frame_pixel_coordinates = warped_feature_binary.nonzero()
    active_frame_pixels_x = np.array(active_frame_pixel_coordinates[1])
    active_frame_pixels_y = np.array(active_frame_pixel_coordinates[0])

    # x-coordinates of the activated frame pixels based on the previous
    # polynomial fitting curves
    left_line_estimated_x = left_line.evaluate_average_polynomial(
        active_frame_pixels_y)
    right_line_estimated_x = right_line.evaluate_average_polynomial(
        active_frame_pixels_y)

    # Active frame pixels falling inside the previous polynomial +/- margin area
    left_line_mask = (
            (active_frame_pixels_x < left_line_estimated_x + margin) &
            (active_frame_pixels_x > left_line_estimated_x - margin))
    right_line_mask = (
            (active_frame_pixels_x < right_line_estimated_x + margin) &
            (active_frame_pixels_x > right_line_estimated_x - margin))

    # Draw the search window into the warped frame based on recent polynomials
    image_height = warped_feature_binary.shape[0]
    y = np.linspace(0, image_height - 1, image_height)
    left_estimated_x = left_line.evaluate_average_polynomial(y)
    right_estimated_x = right_line.evaluate_average_polynomial(y)

    # Create points in a format to draw the polygon
    left_line_left_margin_points = np.column_stack(
        (left_estimated_x - margin, y)).reshape((-1, 1, 2)).astype(int)
    left_line_right_margin_points = np.flipud(np.column_stack(
        (left_estimated_x + margin, y))).reshape((-1, 1, 2)).astype(int)

    right_line_left_margin_points = np.column_stack(
        (right_estimated_x - margin, y)).reshape((-1, 1, 2)).astype(int)
    right_line_right_margin_points = np.flipud(np.column_stack(
        (right_estimated_x + margin, y))).reshape((-1, 1, 2)).astype(int)

    left_vertices = np.vstack((left_line_left_margin_points,
                               left_line_right_margin_points))
    right_vertices = np.vstack((right_line_left_margin_points,
                                right_line_right_margin_points))

    # Draw the search window area onto the warped blank image
    cv2.fillPoly(window_img, [left_vertices], (0, 255, 0))
    cv2.fillPoly(window_img, [right_vertices], (0, 255, 0))

    # Update and fit the lane lines by the new pixel coordinates
    left_line.update(x=active_frame_pixels_x[left_line_mask],
                     y=active_frame_pixels_y[left_line_mask],
                     warped_binary_frame=warped_feature_binary)
    right_line.update(x=active_frame_pixels_x[right_line_mask],
                      y=active_frame_pixels_y[right_line_mask],
                      warped_binary_frame=warped_feature_binary)

    # Color the updated left and right lane line pixels
    if left_line.x is not None and left_line.y is not None:
        colored_lane_lines[left_line.y, left_line.x] = [255, 0, 0]
    if right_line.x is not None and right_line.y is not None:
        colored_lane_lines[right_line.y, right_line.x] = [0, 0, 255]

    # Combine with the colored lane lines image
    window_on_warped_frame = cv2.addWeighted(
        colored_lane_lines, 1, window_img, 0.3, 0)

    # Plot the left and right average polynomials onto the lane lines
    window_on_warped_frame = left_line.draw_polynomial(window_on_warped_frame)
    window_on_warped_frame = right_line.draw_polynomial(window_on_warped_frame)

    return left_line, right_line, window_on_warped_frame


def compute_vehicle_offset(warped_frame_binary,
                           left_line: Line, right_line: Line):
    """ Determine the vehicle offset from the center of the lane.

    It is assumed that the vehicle location is on the bottom center of the
    image. The average horizontal distance of both lane line polynomials
    to the vehicle location is treated as the vehicle's offset to the
    lane center.

    Args:
        warped_frame_binary: warped binary feature of current frame
        left_line: left Line instance
        right_line: right Line instance
    Returns:
          the average offset of the vehicle from the lane center or nan if the
           lines are undetected.
    """

    if left_line.detected and right_line.detected:
        image_height, image_width = warped_frame_binary.shape
        midpoint = image_width // 2  # x-coordinate of image center
        y_vehicle = image_height - 1  # bottom of image = vehicle location
        # Compute the x-coordinate of the lane lines with the average of the
        # recent polynomials
        left_x = left_line.evaluate_average_polynomial(y_vehicle)
        right_x = right_line.evaluate_average_polynomial(y_vehicle)
        # Compute the offset
        offset_pixels = abs(0.5 * (right_x + left_x) - midpoint)
        offset_meters = offset_pixels * Line.xm_per_pix
    else:
        offset_meters = np.nan
    return offset_meters


def post_process(undistorted_frame, features_binary,
                 warped_features, window_img, unwarped_lane,
                 avg_curv_rad, avg_vehicle_offset, left_line, right_line):
    """ Post processing of the current frame.

    Helper function to draw the curvature and offset text on top of the frame.
    Additionally a few small information images are placed on top as well.
    """

    frame_height, frame_width = unwarped_lane.shape[:2]
    ratio = frame_height / frame_width
    x_offset, y_offset = 20, 15
    thumb_width = int((frame_width - 5 * x_offset) / 4)
    thumb_height = int(thumb_width * ratio)
    thumb_dim = (thumb_width, thumb_height)

    top = y_offset
    bot = top + thumb_height

    # ROI thumbnail
    left = x_offset
    right = left + thumb_width
    roi_thumb = cv2.resize(roi_image(undistorted_frame), dsize=thumb_dim)
    unwarped_lane[top:bot, left:right, :] = roi_thumb

    # Features binary
    left = right + x_offset
    right = left + thumb_width
    features_binary_thumb = cv2.resize(features_binary, dsize=thumb_dim)
    features_binary_thumb = np.dstack([features_binary_thumb,
                                       features_binary_thumb,
                                       features_binary_thumb])
    unwarped_lane[top:bot, left:right, :] = features_binary_thumb

    # Warped features
    left = right + x_offset
    right = left + thumb_width
    warped_features_thumb = cv2.resize(warped_features, dsize=thumb_dim)
    warped_features_thumb = np.dstack([warped_features_thumb,
                                       warped_features_thumb,
                                       warped_features_thumb])
    unwarped_lane[top:bot, left:right, :] = warped_features_thumb

    # Lane lines fit
    left = right + x_offset
    right = left + thumb_width
    window_thumb = cv2.resize(window_img, dsize=thumb_dim)
    unwarped_lane[top:bot, left:right, :] = window_thumb

    # Add text for radius of curvature and vehicle offset
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(unwarped_lane,
                f"Average curvature radius: {avg_curv_rad:4.2f} m",
                (350, 220), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(unwarped_lane,
                f"Average vehicle offset: {avg_vehicle_offset:4.2f} m",
                (350, 250),
                font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Add polynomial coefficients text

    # Write polynomial coefficients onto the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    thickness = 1
    A, B, C = left_line.avg_coeff
    cv2.putText(unwarped_lane, f"A={A:+2.1E}",
                (980, 200), font, size, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(unwarped_lane, f"B={B:+2.1E}",
                (980, 220), font, size, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(unwarped_lane, f"C={C:+2.1E}",
                (980, 240), font, size, (255, 255, 255), thickness, cv2.LINE_AA)

    A, B, C = right_line.avg_coeff
    cv2.putText(unwarped_lane, f"A={A:+2.1E}",
                (1140, 200), font, size, (255, 255, 255), thickness,
                cv2.LINE_AA)
    cv2.putText(unwarped_lane, f"B={B:+2.1E}",
                (1140, 220), font, size, (255, 255, 255), thickness,
                cv2.LINE_AA)
    cv2.putText(unwarped_lane, f"C={C:+2.1E}",
                (1140, 240), font, size, (255, 255, 255), thickness,
                cv2.LINE_AA)

    return unwarped_lane
