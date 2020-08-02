import numpy as np
import cv2
import pickle

# Load camera matrix and distortion coefficients
calibration_results = pickle.load(open("calibration_results.p", "rb"))
mtx = calibration_results["mtx"]
dist = calibration_results["dist"]

# Region of interest, used as perspective transform source
# in ratios of the image width (x) and height (y)
# Note: Optimized for the undistorted images!
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


def roi_image(undistorted_rgb_img):
    # ROI polygon vertices
    x, y = undistorted_rgb_img.shape[1], undistorted_rgb_img.shape[0]
    vertices = np.array([[(ROI_SRC["x1"] * x, ROI_SRC["y1"] * y),
                          (ROI_SRC["x2"] * x, ROI_SRC["y2"] * y),
                          (ROI_SRC["x3"] * x, ROI_SRC["y3"] * y),
                          (ROI_SRC["x4"] * x, ROI_SRC["y4"] * y)]],
                        dtype=int)

    # Intilialize mask
    mask = np.zeros_like(undistorted_rgb_img)

    # Color channels count depends on the input image
    if len(undistorted_rgb_img.shape) > 2:
        channel_count = undistorted_rgb_img.shape[2]
        active_mask = (255,) * channel_count
    else:
        active_mask = 255

    # Create the polygon mask
    cv2.fillPoly(mask, vertices, active_mask)

    # Black out all pixels of the image where the mask has zero values
    masked_image = cv2.bitwise_and(undistorted_rgb_img, mask)

    # Combine original and roi mask image
    return cv2.addWeighted(undistorted_rgb_img, 0.3, masked_image, 0.7, 0.)


def rgb2gray(rgb_img):
    """ Convert an image from rgb to gray.

    Args:
        rgb_img: The rgb image
    """
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)


def rgb2hsv(rgb_img, channel=None):
    """ Convert an image from rgb to hsv.

    Args:
        rgb_img: The rgb image
        channel: the hsv channel that shall be returned, defaults to all
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv if channel is None else hsv[:, :, channel]


def rgb2hls(rgb_img, channel=None):
    """ Convert an image from rgb to hls.

    Args:
        rgb_img: The rgb image
        channel: the hls channel that shall be returned, defaults to all
    """
    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    return hls if channel is None else hls[:, :, channel]


def undistort(image):
    return cv2.undistort(image, mtx, dist)


def rgb2clahe(rgb_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Perform a contrast limited adaptive histogram equalization on given image.

    The clahe operation is performed on the grayscale version of the given rgb
    frame.

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


def sobel(img_channel, thresh_min=20, thresh_max=255, ksize=9):
    """ Determine and threshold the sobel magnitudes on a single channel image.

    Args:
        img_channel: single channel image
        thresh_min: lower threshold for the sobel magnitude
        thresh_max: upper threshold for the sobel magnitude
        ksize: kernel size for cv2.Sobel
    Returns:
          a binary mask with 255 values where the threshold conditions are
           fullfilled
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


def sobel_feature(rgb_image, thresh_min=20, thresh_max=255, ksize=9,
                  iterations=5):
    """
    Create a mask by thresholding the sobel magnitude in the HLS color space.

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
