import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

from features import ROI_SRC, ROI_DST, undistort, feature_mask
from process import fit_lines_by_windows, fit_lines_by_recent_lines, warp
from process import draw_lane_on_unwarped_frame, compute_vehicle_offset
from process import Line, post_process

# Global variables
left_line = Line(cache=20)
right_line = Line(cache=20)
frame_counter = 0


def process(frame):
    """ The main processing function.

    Each frame undergoes the complete pipeline.

    Args:
        frame: the current raw video frame
    Returns:
        the processed frame
    """

    global left_line, right_line, frame_counter

    # Undistort the current frame
    undistorted_frame = undistort(frame)

    # Create the binary feature mask for the undistorted frame
    features_binary = feature_mask(undistorted_frame)

    # Perspective transformation (warp) of the features binary
    warped_features, trsf_mtx, trsf_mtx_inv = warp(
        features_binary, ROI_SRC, ROI_DST)

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

    # if frame_counter % 30 == 0:
        # mpimg.imsave(f"test/project_video_{frame_counter}.jpg", frame)
        # mpimg.imsave(f"test/challange_{frame_counter}.jpg", frame)
        # mpimg.imsave(f"test/harder_challange_{frame_counter}.jpg", frame)

    frame_counter += 1

    return processed_frame


if __name__ == "__main__":

    video_mode = True
    # video_mode = False

    if video_mode:

        # video = "project_video.mp4"
        video = "challenge_video.mp4"
        # video = "harder_challenge_video.mp4"

        output = f"output_videos/{video}"
        # To speed up the testing process you may want to try your pipeline
        # on a shorter subclip of the video.
        # To do so add .subclip(start_second,end_second) to the end of the
        # line below where start_second and end_second are integer values
        # representing the start and end of the subclip.
        # clip1 = VideoFileClip(f"{video}").subclip(0, 5)
        clip1 = VideoFileClip(f"{video}")
        # NOTE: this function expects color images!!
        clip = clip1.fl_image(process)
        clip.write_videofile(output, audio=False)

    else:

        # TEST_DIR = "test_images"
        # OUTPUT_DIR = "output_images"
        TEST_DIR = "test_images_extended/challange"
        OUTPUT_DIR = "./output_images_extended"
        files = [file for file in os.listdir(TEST_DIR)]

        for i, file in enumerate(files):
            frame = mpimg.imread(f"{TEST_DIR}/{file}")
            processed_frame = process(frame)
            mpimg.imsave(f"{OUTPUT_DIR}/{i}_processed_{file}", processed_frame)
            left_line = Line()
            right_line = Line()
