import numpy as np
import cv2
import pickle

from pathlib import Path

# Creation of the array of evenly spaced 9x6 object corner points (x, y, z)
cols = 9  # x-axis, respectively u-axis
rows = 6  # y-axis, respectively v-axis
object_corners = np.zeros((rows * cols, 3)).astype(np.float32)
# Fill the array with the x and y coordinates. The z-coordinate stays 0.
object_corners[:, :2] = np.mgrid[0: cols, 0: rows].T.reshape(-1, 2)
# >>> object_corners
# array([[0., 0., 0.],
#        [1., 0., 0.],
#            ...
#        [7., 5., 0.],
#        [8., 5., 0.]])

# Initialize colletors for the object and image points of all images
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

        # Draw corners
        # cv2.drawChessboardCorners(img, (cols, rows), image_corners, ret)
        # cv2.imwrite(f"output_images/{file.stem}_corners.jpg", img)

# Camera calibration (point coordinates must be type float32)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_corner_list, image_corner_list, gray.shape[::-1], None, None)

# Store the camera matrix and the distortion coefficients for later use
pickle.dump({"mtx": mtx, "dist": dist}, open("calibration_results.p", "wb"))

