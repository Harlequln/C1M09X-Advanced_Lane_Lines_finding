This project is part of Udacity's [Self-Driving-Car Nanodegree][Course]. The project
resources and build instructions can be found [here][Project].

## Lane lines detection using computer vision
The main goal of this project is the development of a software pipeline to detect the lane 
boundaries in the provided videos. The pipeline can be subdividided into the following 
steps:

1. One-time determination of the camera calibration matrix and its distortion coefficients.
2. Use it to correct the distortion of each video frame.
3. Create a binary feature mask to detect the lane lines.
4. Transform the binary image into bird's-eye perspective.
5. Detecting and fitting the left and right lane lines to second order polynomials.
6. Determination of average radius of curvature and vehicle offset with respect to  
   lane center.
7. Visualization of lane boundaries and additional information on each video frame. 

## Camera calibration
The required code for the camera calibration can be found in [``camera_calibration.py``][Calibration]. 
There is also a nice [OpenCV tutorial][CalTut] on this topic.

Chessboard sample images were taken from different angles with the embedded camera and 
can be found in the [``camera_cal``][Chessboard] folder. On most of them A 9x6 pattern can 
be seen. To determine the camera calibration matrix, two sets of points must be compared:

- the *object points* of the chessboard corners in the 3D world (X, Y, Z), and
- the corresponding 2D chessboard corner *image points* (X, Y)

For simplicity, it is assumed that the 3D object points lie on the XY-plane and the 
Z-coordinate is zero.

```python
# Creation of the array of evenly spaced 9x6 object points (x, y, z)
cols = 9  # x-axis
rows = 6  # y-axis
object_corners = np.zeros((rows * cols, 3)).astype(np.float32)

# Fill the array with the x and y coordinates. The z-coordinate stays 0.
object_corners[:, :2] = np.mgrid[0: cols, 0: rows].T.reshape(-1, 2)

# >>> object_corners
# array([[0., 0., 0.],
#        [1., 0., 0.],
#            ...
#        [7., 5., 0.],
#        [8., 5., 0.]])
```

The corresponding 2D image points are determined by using the 
[``cv2.findChessboardCorners``][findChessboard] function after the chessboard image has 
been converted to grayscale.

```python
object_corner_list = []  # corner 3-d coordinates in the world space
image_corner_list = []  # corner 2-d coodinates in the image

for file in Path("camera_cal/").rglob("*.jpg"):

    # Preprocess images
    img = cv2.imread(str(file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, image_corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    # Collect corner coordinates if found
    if ret is True:
        object_corner_list.append(object_corners)
        image_corner_list.append(image_corners)
```

The camera calibration can now be perfomed with [``cv2.calibrateCamera``][calibrateCamera] 
using the obtained image and object points. The same object points are used for the 
comparison with the detected image points on all chessboard images.

```python
# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_corner_list, image_corner_list, gray.shape[::-1], None, None)

# Store the camera matrix and the distortion coefficients for later use
pickle.dump({"mtx": mtx, "dist": dist}, open("calibration_results.p", "wb"))
```

From now on, images taken with this camera can be undistorted using 
[``cv2.undistort``][undistort] with the camera matrix ``mtx`` and the distortion 
coefficients ``dist``.

```python
# Undistort one of the calibration images
img = mpimg.imread("camera_cal/calibration1.jpg")
undist_img = cv2.undistort(img, mtx, dist, dst=None, newCameraMatrix=mtx)
mpimg.imsave(f"output_images/calibration1_undistorted.jpg", undist_img)
plot_undistort_example(img, undist_img, save="./output_images/camera_calibration_example")
```
![alt text][image1]

## Undistortion of current video frame
Each new video frame must be undistorted before it can be further processed. 
This can be done exactly as in the camera calibration example above

```python
# Undistort one of the test images
img = mpimg.imread("./test_images/test1.jpg")
undist_img = cv2.undistort(img, mtx, dist, dst=None, newCameraMatrix=mtx)
mpimg.imsave(f"output_images/test1_undistorted.jpg", undist_img)
plot_undistort_example(img, undist_img, save="./output_images/undistort_example")
```
![alt text][image2]


## Binary feature mask for lane detection
This is one of the major steps of the project. A binary feature mask needs to be created 
for the subsequent lane line detection in each undistorted video frame. Three features were
chosen to accomplish this task:

- The [``white_feature``][white_feature], whose major purpose is to detect the white lane 
  lines
- The [``yellow_feature``][yellow_feature], whose major purpose is to detect the yellow lane
  lines
- The [``sobel_feature``][sobel_feature] which provides additional edge information about 
  both lane lines

The complete feature code including all helper functions can be found in [``features.py``][features].

### The white feature
The white feature uses the concept of contrast limited adaptive histogram equalization 
([CLAHE][CLAHE]). After performing the histogram equalization for the RGB frame with the 
[``rgb2clahe``][rgb2clahe] helper function, the resulting image is thresholded to obtain 
the white feature mask.

```python
def white_feature(rgb_image, thresh_min=210):
    """ Create a mask for white lane lines detection.

    A contrast limited adaptive histogram equalization is peformed on the
    given frame and the resulting gray image is subsequently thresholded by 
    thresh_min.

    Args:
        rgb_image: current undistorted rgb frame
        thresh_min: lower threshold for the gray clahe image
    Returns:
        a binary mask with 255 values where white lines are assumed.
    """
    clahe = rgb2clahe(rgb_image)
    mask = np.zeros_like(clahe)
    mask[(clahe > thresh_min)] = 255
    return mask
```

```python
def rgb2clahe(rgb_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Perform a contrast limited adaptive histogram equalization on given image.

    The clahe operation is performed on the grayscale version of the given rgb frame.

    Args:
        rgb_img: current undistorted rgb frame
        clip_limit: threshold for contrast limiting
        tile_grid_size: size of the grid for the histogram equalization.
         The image will be divided into equally sized rectangular tiles.
         tile_grid_size defines the number of tiles in row and column.
    Returns:
        a gray image as result of the application of clahe
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_clahe = clahe.apply(gray)
    return gray_clahe
```

The lower threshold ``thresh_min=210`` was chosen as default after some tuning iterations. 

![alt text][image3]

### The yellow feature
The yellow feature uses color thresholding in the HSV color space to detect yellow color 
in the given rgb frame. 

```python
def yellow_feature(rgb_image, lbound=(17, 20, 140), ubound=(32, 255, 255)):
    """  Create a mask for yellow lane lines detection.

    The yellow lines are detected by converting the rgb_image into hsv space
    and thesholding the yellow color.

    Args:
        rgb_image: current undistorted rgb frame
        lbound: lower hsv bounds
        ubound: upper hsv bounds
    Returns:
        a binary mask with 255 values where yellow lines are assumed.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    ylw_mask = cv2.inRange(hsv, lbound, ubound)
    return ylw_mask
```

The thresholds of each channel were tuned to preserve only the yellow color.
![alt text][image4]
![alt text][image5]
![alt text][image6]

### The sobel feature
The sobel feature creates a binary mask by determining and thresholding the sobel 
magnitude on the saturation and lightness channels of the HLS color space. A morphological 
closure operation is then performed on the binary mask.

```python
def sobel_feature(rgb_image, thresh_min=20, thresh_max=255, ksize=9, iterations=5):
    """ Create a mask by thresholding the sobel magnitude in the HLS color space.

    The sobel magnitude is determined on the saturation and lightness channels.
    A binary mask is created by thresholding the magnitudes in between
    thresh_min and thresh_max. The composition of both binary masks finally
    undergoes a morphological closing operation by the given iterations.

    Args:
        rgb_image: current undistorted rgb frame
        thresh_min: lower threshold for the sobel magnitude
        thresh_max: upper threshold for the sobel magnitude
        ksize: kernel size for cv2.Sobel
        iterations: iterations for morphological closing
    Returns:
          binary mask with 255 values at actice pixels
    """
    s_img = rgb2hls(rgb_image, channel=2)
    l_img = rgb2hls(rgb_image, channel=1)
    s_mask = sobel(s_img, thresh_min, thresh_max, ksize)
    l_mask = sobel(l_img, thresh_min, thresh_max, ksize)
    sobel_mask = cv2.bitwise_or(s_mask, l_mask)
    mask = cv2.morphologyEx(sobel_mask, cv2.MORPH_CLOSE, kernel=np.ones((3, 3)),
                            iterations=iterations)
    return mask
```

```python
def sobel(img_channel, thresh_min=20, thresh_max=255, ksize=9):
    """ Determine and threshold the sobel magnitudes on a single channel image.

    Args:
        img_channel: single channel image
        thresh_min: lower threshold for the sobel magnitude
        thresh_max: upper threshold for the sobel magnitude
        ksize: kernel size for cv2.Sobel
    Returns:
          a binary mask with 255 values where the threshold condition is fullfilled
    """
    # Sobel gradients
    sobel_x = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=ksize)

    # Magnitude
    sobel_mag = np.sqrt(sobel_x ** 2, sobel_y ** 2)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    mag_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    # Mask of 255's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    mask = np.zeros_like(mag_scaled)
    mask[(mag_scaled > thresh_min) & (mag_scaled < thresh_max)] = 255

    return mask
```
```python
def rgb2hls(rgb_img, channel=None):
    """ Convert an image from rgb to hls.

    Args:
        rgb_img: rgb image
        channel: the hls channel that shall be returned, defaults to all
    """
    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    return hls if channel is None else hls[:, :, channel]
```

The lightness and saturation channels were chosen because they work well together 
and are both "activated" on the white and yellow lane lines.

![alt text][image7]
![alt text][image8]

### Feature composition
The final feature mask is created by combining all three subfeatures. It only keeps 
pixels active if at least two of the three subfeatures are active.

```python
def feature_mask(rgb_image):
    """ Main lane line detection feature.

    The combination of white_feature, yellow_feature and sobel_feature.

    Args:
        rgb_image: current undistorted rgb frame
    Returns:
        A mask for the lane line detection
    """
    white_mask = white_feature(rgb_image).astype(float)
    yellow_mask = yellow_feature(rgb_image).astype(float)
    sobel_mask = sobel_feature(rgb_image).astype(float)

    # Equal weighted mask
    mask = (white_mask + yellow_mask + sobel_mask) / 255 >= 2

    return np.where(mask, 255, 0).astype(np.uint8)
```
![alt text][image9]

## Transformation into birds-eye perspective
To perform the perspective transformation, four points are required for the mapping of 
the source locations to the destination locations. In this approach, the points are defined 
as ratios of image size. The source points additionally serve for the defintion of the 
region of interest in front of the vehicle.  

```python
# Region of interest, used as perspective transform source
# in ratios of the image width (x) and height (y)
ROI_SRC = {"x1": 0.03, "y1": 0.98,  # lower left corner
           "x2": 1.00, "y2": 0.98,  # lower right corner
           "x3": 0.59, "y3": 0.65,  # upper right corner
           "x4": 0.42, "y4": 0.65}  # upper left corner

# Perspective transform destination
# in ratios of the image width (x) and height (y)
ROI_DST = {"x1": 0.0, "y1": 1.0,  # left lower corner
           "x2": 1.0, "y2": 1.0,  # right lower corner
           "x3": 1.0, "y3": 0.0,  # right upper corner
           "x4": 0.0, "y4": 0.0}  # left upper corner
```

The [``warp``][warp] function in [``process.py``][process] performs the perspective 
transformation on the given undistorted image. It returns the transformed image and 
additionally the transformation matrix and its inverse for the back transformation, 
which is performed in a later step. 

```python
def warp(img, roi_src, roi_dst):
    """ Perform a perspective transform on the given undistorted image.

    Args:
        img: undistorted image of current frame, e.g. the binary feature mask
        roi_src: perspective transform source as ratios for image dimensions x and y
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
```
In the main pipeline, the binary feature mask is transformed into bird's-eye view. 
The following image is only for illustration of the warping process.

![alt text][image10]

## Lane lines detection
The part of the main pipeline dealing with lane detection can be described by the 
following steps:

1. Undistortion of the current RGB frame
2. Determination of the binary feature mask on the undistorted image
3. Warping of the binary feature mask into birds-eye view
4. Detection of the left and right lane lines and fitting to a second order polynomial  
    4.1. by a sliding window approach (if previously undetected or in lost status) or   
    4.2. by using the information of recently detected lane lines

Both functions [``fit_lines_by_windows``][fit_lines_by_windows] and 
[``fit_lines_by_recent_lines``][fit_lines_by_recent_lines] can be found in 
[``process.py``][process].

```python
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
```

```python
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
```

The [``Line``][Line] class is used to collect the required information for each lane line. 
It is updated by the active pixels determined by [``fit_lines_by_windows``][fit_lines_by_windows] 
or [``fit_lines_by_recent_lines``][fit_lines_by_recent_lines] on 
each new incoming video frame. Among others, the following steps are performed:

- Sanity check of new polynomial fit (coefficient deviations too high?)
- Depending on the sanity check, setting to lost or detected status
- Update of the current radius of curvature
- Update of the current offset of the line from the vehicle location

The complete [``Line``][Line] class code can be found in [``process.py``][process]. 

```python
class Line:
    """ Gather and update lane lines information. """

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
```

![alt text][image11]

## Radius of curvature and vehicle offset
Both values are computed by taking the average of the values of each individual lane line. 
This is a step in the main processing function [``process``][process_func] in 
[``advanced_lane_finding.py``][advanced_lane_finding]. 

```python
# Compute the vehicle offset and the average curvature radius 
avg_vehicle_offset = 0.5 * abs(left_line.offset_from_vehicle + right_line.offset_from_vehicle)
avg_curv_rad = 0.5 * (left_line.radius_of_curvature + right_line.radius_of_curvature)
```

The actual computation of the individual lane line values is done by the methods 
[``Line.update_offset_from_vehicle``][update_offset] and 
[``Line.update_radius_of_curvature_m``][update_radius]. 
It is assumed that the vehicle location is at the bottom center of the image. The vehicle's
offset to the lane center is computed by taking the average horizontal distance of both lane
line polynomials to the vehicle location. The formula for the radius of curvature can be 
found [here][link2].
 

```python
class Line:

    # Convertion of image to real world dimensions
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

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
```

## Visualization
The complete processing pipeline is performed on each incoming video frame by passing it to 
the [``process``][process_func] function in [``advanced_lane_finding.py``][advanced_lane_finding].

```python
def process(frame):
    """ The main processing function.

    Each frame undergoes the complete pipeline.

    Args:
        frame: the current raw video frame
    Returns:
        the processed frame
    """

    global left_line, right_line

    # Undistort the current frame
    undistorted_frame = undistort(frame)

    # Create the binary feature mask for the undistorted frame
    features_binary = feature_mask(undistorted_frame)

    # Perspective transformation (warp) of the features binary
    warped_features, trsf_mtx, trsf_mtx_inv = warp(features_binary, ROI_SRC, ROI_DST)

    # Detect and refit the lane lines on the warped features binary
    if not left_line.detected or not right_line.detected:
        # If the lines were not detected before or are in lost status relocate
        # by the sliding windows approach
        left_line, right_line, window_img = fit_lines_by_windows(
            warped_features, left_line, right_line)
    else:
        # If the lines are still detected, the recent lines information will be
        # used for the current detection
        left_line, right_line, window_img = fit_lines_by_recent_lines(
            warped_features, left_line, right_line)

    # Draw the detected lane lines and lane area on the unwarped frame
    unwarped_lane = draw_lane_on_unwarped_frame(
        undistorted_frame, left_line, right_line, trsf_mtx_inv)

    # Compute the vehicle offset and the average curvature radius 
    avg_vehicle_offset = 0.5 * abs(left_line.offset_from_vehicle
                                   + right_line.offset_from_vehicle)
    avg_curv_rad = 0.5 * (left_line.radius_of_curvature
                          + right_line.radius_of_curvature)

    # Draw intermediate results on top of the main frame for information
    processed_frame = post_process(undistorted_frame, features_binary,
                                   warped_features, window_img, unwarped_lane,
                                   avg_curv_rad, avg_vehicle_offset,
                                   left_line, right_line)
    
    return processed_frame
```

The processed [``project_video.mp4``][projvid] and [``challenge_video.mp4``][chalvid] can be found in the 
[``output_videos``][output_videos] folder.

#### Processed project video
![][image13]

#### Processed challenge video
![][image14]

## Discussion
The approach presented above works well for the ``project_video.mp4`` and the ``challenge_video.mp4``. 
Of course, this is also due to the fact that it was developed on basis of test images taken from these 
two videos. There is certainly room for improvement, especially in cases where the lane lines are lost 
over several frames.   
Shorter periods could be bridged in this approach by giving the Line classes a memory of 20 frames. 
By averaging the lane line polynomials, extreme jumps in individual frames could be reduced. 
Additionally, the limitation of possible deviations of the polynomial coefficients A and C, also serves 
this purpose. If, due to a lack of information in individual frames, these parameters 
are about to change strongly in comparison to the previous history, the line status is set to lost and 
navigation is continued based on the recent history. However, if the lost status exceeds a certain number 
of frames, the history is deleted and an attempt is made to get back on track with a complete reset. 
This method certainly has a lot of optimization potential: 

- How big should the memory cache be? 
- Number of lost status frames before reset?
- How to make the reset behave *soft*?  
- Which parameters to use to measrure the lane lines quality? 

Other critical situations are certainly also:

- Sharp curves with lanes running out of the visible image
- Shadows and glare
- Dirty roads
- Concealed lane lines
- Overtaking cars crossing the own lane

[Course]: https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
[Project]: https://github.com/udacity/CarND-Advanced-Lane-Lines
[Chessboard]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/tree/master/camera_cal
[Calibration]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/calibration1_undistorted.jpg
[CalTut]: https://docs.opencv.org/5.x/dc/dbb/tutorial_py_calibration.html
[findChessboard]: https://docs.opencv.org/5.x/d4/d93/group__calib.html#ga93efa9b0aa890de240ca32b11253dd4a
[calibrateCamera]: https://docs.opencv.org/5.x/d4/d93/group__calib.html#gae4d3c8c61e181b222921991fc6a583ca
[undistort]: https://docs.opencv.org/5.x/da/d35/group____3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d
[Features]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/features.py
[white_feature]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/features.py#L112
[yellow_feature]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/features.py#L131
[sobel_feature]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/features.py#L179
[rgb2clahe]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/features.py#L90
[warp]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/process.py#L178
[process]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/process.py
[fit_lines_by_windows]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/process.py#L270
[fit_lines_by_recent_lines]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/process.py#L417
[Line]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/process.py#L9
[process_func]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/advanced_lane_finding.py#L18
[advanced_lane_finding]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/advanced_lane_finding.py
[update_offset]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/process.py#L115
[update_radius]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/blob/master/process.py#L139
[output_videos]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/tree/master/output_videos
[projvid]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/tree/master/output_videos/project_video.mp4
[chalvid]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/tree/master/output_videos/challenge_video.mp4

[image1]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/camera_calibration_example.png "Undistortion Example"
[image2]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/undistort_example.png "Undistortion Example"
[image3]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/white_feature_tuning.png "White feature tuning"
[image4]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/yellow_feature_h_tuning.png "Yellow feature hue tuning"
[image5]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/yellow_feature_s_tuning.png "Yellow feature saturation tuning"
[image6]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/yellow_feature_v_tuning.png "Yellow feature value tuning"
[image7]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/sobel_feature_thresh_min_tuning.png "Sobel feature thresh_min tuning"
[image8]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/sobel_feature_iterations_tuning.png "Sobel feature iterations tuning"
[image9]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/feature_composition.png "Feature composition"
[image10]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/perspective_transform.png "Perspective transformation"
[image11]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/lane_detection_pipeline.png "Lane line detection pipeline"
[image12]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_images/test2_processed.jpg "Example pipeline output"
[image13]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_videos/project_video_640.gif "Processed project video"
[image14]: https://github.com/pabaq/CarND-Advanced-Lane-Finding/raw/master/output_videos/challenge_video_640.gif "Processed challenge video"

[CLAHE]: https://docs.opencv.org/5.x/d5/daf/tutorial_py_histogram_equalization.html
[Link2]: https://www.intmath.com/applications-differentiation/8-radius-curvature.php
