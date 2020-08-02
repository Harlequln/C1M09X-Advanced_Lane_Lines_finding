import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from features import *
from process import *
from advanced_lane_finding import *


def plot_undistort_example(img, undist_img,
                           save="./output_images/undistort_example"):
    """ Creation of the undistortion demonstration image for the writeup. """

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].imshow(undist_img)
    axes[1].set_title("Undistorted Image")
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    fig.savefig(save)


def plot_warped_example(img, warped_img,
                        save="./output_images/perspective_transform_example"):
    """ Creation of the perspective transformation image for the writeup. """

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].imshow(warped_img)
    axes[1].set_title("Warped Image")
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    fig.savefig(save)


def plot_process_example(img, save="./output_images/process_example"):
    """ Creation of the processed image for the writeup. """

    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.suptitle("Processing Output", size=20)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(save)


def abs_sobel_thresh(img_channel, orient='x', thresh_min=0, thresh_max=255,
                     ksize=9):
    # Sobel gradiants
    sobel_x = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=ksize)

    # Absolute value of gradients
    sobel_x_abs = np.abs(sobel_x)
    sobel_y_abs = np.abs(sobel_y)
    sobel_mag = np.sqrt(sobel_x ** 2, sobel_y ** 2)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_x_scaled = np.uint8(255 * sobel_x_abs / np.max(sobel_x_abs))
    sobel_y_scaled = np.uint8(255 * sobel_y_abs / np.max(sobel_y_abs))
    sobel_mag_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    if orient == "x":
        sobel = sobel_x_scaled
    elif orient == "y":
        sobel = sobel_y_scaled
    else:
        sobel = sobel_mag_scaled

    # Mask of 255's where the scaled gradient magnitude
    mask = np.zeros_like(sobel)
    # is > thresh_min and < thresh_max
    mask[(sobel > thresh_min) & (sobel < thresh_max)] = 255
    # 6) Return this mask as your binary_output image
    return mask


def plot_perspective_transform(rgb_sample, mtx, dist, src, dst):
    rows = 2
    cols = len(rgb_sample)
    y, x, c = rgb_sample[0].shape
    width = 15

    fig_size = (cols * width * y / x, rows * width * 0.45)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fs = round(fig_size[0]) * 1.5

    for col, rgb_img in enumerate(rgb_sample):
        # Undistort image
        undist_img = cv2.undistort(rgb_img, mtx, dist)

        # Create image highlighting the roi area
        rgb_roi_mask = roi_image(undist_img)
        rgb_roi_img = cv2.addWeighted(undist_img, 0.3, rgb_roi_mask, 0.7, 0.)

        # Create warped image
        rgb_warped_img, _, _ = warp(undist_img, src, dst)

        axes[0, col].imshow(rgb_roi_img)
        axes[0, col].get_xaxis().set_visible(False)
        axes[0, col].get_yaxis().set_visible(False)
        axes[1, col].imshow(rgb_warped_img)
        axes[1, col].get_xaxis().set_visible(False)
        axes[1, col].get_yaxis().set_visible(False)

    plt.tight_layout()
    fig.suptitle("Perspective transformation of the region of interest",
                 size=fs)
    fig.savefig("./output_images/perspective_transform")


def plot_channels(rgb_sample, orient=None, thresh_min=0, thresh_max=255,
                  ksize=3):
    """ Plot RGB, HSV and HLS channels for a rgb image sample. """

    # Grid variables
    rows = len(rgb_sample)
    cols = 11
    y, x, c = rgb_sample[0].shape
    width = 8

    fig_size = (cols * width, rows * width * y / x)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fs = round(fig_size[0]) / 2

    for row in range(rows):
        rgb_img = rgb_sample[row]
        gray_img = rgb2gray(rgb_img)
        hsv_img = rgb2hsv(rgb_img)
        hls_img = rgb2hls(rgb_img)

        # Plot rgb image
        ax = axes[row, 0]
        ax.imshow(rgb_img)

        # Plot gray image
        ax = axes[row, 1]
        if orient is None:
            ax.imshow(gray_img, cmap='gray')
        else:
            ax.imshow(abs_sobel_thresh(
                gray_img, orient, thresh_min, thresh_max, ksize), cmap='gray')

        image = {"rgb": rgb_img, "hsv": hsv_img, "hls": hls_img}
        offset = {"rgb": 2, "hsv": 5, "hls": 8}
        for space, img in image.items():
            for channel in [0, 1, 2]:
                ax = axes[row, channel + offset[space]]
                if orient is None:
                    if channel == 0 and space in ["hsv", "hls"]:
                        c = ax.imshow(img[:, :, channel], vmin=0, vmax=180)
                    else:
                        c = ax.imshow(img[:, :, channel], vmin=0, vmax=255)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    fig.colorbar(c, cax=cax)
                else:
                    ax.imshow(
                        abs_sobel_thresh(
                            img[:, :, channel],
                            orient, thresh_min, thresh_max, ksize),
                        cmap="gray")

    axes[0, 0].set_title("rgb", size=fs)
    axes[0, 1].set_title("gray", size=fs)
    axes[0, 2].set_title("R\nRed", size=fs)
    axes[0, 3].set_title("G\nGreen", size=fs)
    axes[0, 4].set_title("B\nBlue", size=fs)
    axes[0, 5].set_title("H\nHue", size=fs)
    axes[0, 6].set_title("S\nSaturation", size=fs)
    axes[0, 7].set_title("V\nValue", size=fs)
    axes[0, 8].set_title("H\nHue", size=fs)
    axes[0, 9].set_title("L\nLightness", size=fs)
    axes[0, 10].set_title("S\nSaturation", size=fs)

    if orient is not None:
        fig.suptitle(f"Sobel Magnitude "
                     f"(thres_min={thresh_min}, thres_max={thresh_max}, "
                     f"kernel_size={ksize})", size=fs * 1.5)
    fig.savefig("output_images/channels") if orient is None else fig.savefig(
        f"output_images/abs_sobel_{orient}_{thresh_min}_{thresh_max}_{ksize}")


def plot_white_feature(rgb_sample):
    # Grid variables
    rows = len(rgb_sample)
    cols = 7
    y, x, c = rgb_sample[0].shape
    width = 8

    fig_size = (cols * width, rows * width * y / x * 0.85)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fs = round(fig_size[0]) / 2

    for row in range(rows):
        rgb_img = rgb_sample[row]

        ax = axes[row, 0]
        ax.imshow(rgb_img)

        ax = axes[row, 1]
        c = ax.imshow(rgb2clahe(rgb_img), cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(c, cax=cax)

        for i, thresh_min in enumerate(np.linspace(200, 240, 5)):
            col = i + 2
            ax = axes[row, col]
            white_mask = white_feature(rgb_img, thresh_min=thresh_min)
            ax.imshow(white_mask, cmap="gray")
            if row == 0:
                color = "r" if thresh_min == 210 else "k"
                axes[row, col].set_title(
                    f"white_feature(\n"
                    f"rgb_img,\n"
                    f"thresh_min={thresh_min})",
                    size=fs, color=color)

    axes[0, 0].set_title("rgb_img", size=fs)
    axes[0, 1].set_title(
        f"cv2.createCLAHE(\nclipLimit=2, tileGridSize=(8, 8))",
        size=fs)

    fig.suptitle(f"White feature tuning", size=fs * 2)
    fig.savefig("output_images/white_feature_tuning")


def plot_yellow_feature(rgb_sample, tuning="h"):
    # Grid variables
    rows = len(rgb_sample)
    cols = 8
    y, x, c = rgb_sample[0].shape
    width = 8

    fig_size = (cols * width, rows * width * y / x * 0.9)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fs = round(fig_size[0]) / 2

    lbounds_s = [(17, 0, 140), (17, 10, 140), (17, 20, 140), (17, 30, 140)]
    ubounds_s = [(32, 255, 255), (32, 255, 255), (32, 255, 255), (32, 255, 255)]

    lbounds_h = [(15, 20, 140), (17, 20, 140), (19, 20, 140), (21, 20, 140)]
    ubounds_h = [(32, 255, 255), (32, 255, 255), (32, 255, 255), (32, 255, 255)]

    lbounds_v = [(17, 20, 110), (17, 20, 140), (17, 20, 170), (17, 20, 200)]
    ubounds_v = [(32, 255, 255), (32, 255, 255), (32, 255, 255), (32, 255, 255)]

    if tuning == "h":
        lbounds = lbounds_h
        ubounds = ubounds_h
    elif tuning == "s":
        lbounds = lbounds_s
        ubounds = ubounds_s
    elif tuning == "v":
        lbounds = lbounds_v
        ubounds = ubounds_v

    for row in range(rows):
        rgb_img = rgb_sample[row]

        ax = axes[row, 0]
        ax.imshow(rgb_img)

        for channel in [0, 1, 2]:
            col = channel + 1
            ax = axes[row, col]
            vmax = 180 if channel == 0 else 255
            c = ax.imshow(rgb2hsv(rgb_img, channel=channel), vmin=0, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(c, cax=cax)

        for i, (lbound, ubound) in enumerate(zip(lbounds, ubounds)):
            col = i + 4
            ax = axes[row, col]
            ylw_mask = yellow_feature(rgb_img, lbound, ubound)
            ylw_mask_rgb = cv2.bitwise_and(rgb_img, rgb_img, mask=ylw_mask)
            ax.imshow(ylw_mask_rgb)
            if row == 0:
                h_lb = lbound[0]
                s_lb = lbound[1]
                v_lb = lbound[2]
                if h_lb == 17 and tuning == "h":
                    color = "r"
                elif s_lb == 20 and tuning == "s":
                    color = "r"
                elif v_lb == 140 and tuning == "v":
                    color = "r"
                else:
                    color = "k"

                axes[row, col].set_title(
                    f"yellow_feature(\n"
                    f"rgb_img,\n"
                    f"lbound={lbound},\n"
                    f"ubound={ubound})",
                    size=fs, color=color)

    axes[0, 0].set_title("rgb_img", size=fs)
    axes[0, 1].set_title("Hue", size=fs)
    axes[0, 2].set_title("Saturation", size=fs)
    axes[0, 3].set_title("Value", size=fs)
    title = {"h": "Hue", "s": "Saturation", "v": "Value"}
    fig.suptitle(f"{title[tuning]} lower bound tuning", size=fs * 2)
    fig.savefig(f"output_images/yellow_feature_{tuning}_tuning")


def plot_sobel_feature(rgb_sample, tuning="iterations"):
    # Grid variables
    rows = len(rgb_sample)
    cols = 8
    y, x, c = rgb_sample[0].shape
    width = 8

    fig_size = (cols * width, rows * width * y / x)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fs = round(fig_size[0]) / 2

    pars = []
    if tuning == "iterations":
        pars = [1, 3, 5, 7]
    elif tuning == "thresh_min":
        pars = [0, 20, 40, 60]

    for row in range(rows):
        rgb_img = rgb_sample[row]

        ax = axes[row, 0]
        ax.imshow(rgb_img)

        for channel in [0, 1, 2]:
            col = channel + 1
            ax = axes[row, col]
            vmax = 180 if channel == 0 else 255
            c = ax.imshow(rgb2hls(rgb_img, channel=channel), vmin=0, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(c, cax=cax)

        for i in range(len(pars)):
            kwargs = {tuning: pars[i]}
            col = i + 4
            ax = axes[row, col]
            sobel_s_mask = sobel_feature(rgb_img, **kwargs)
            ax.imshow(sobel_s_mask, cmap="gray")
            if row == 0:
                color = "r" if kwargs[tuning] in [5, 20] else "k"
                thresh_min = 20 if tuning != "thresh_min" else pars[i]
                iterations = 5 if tuning != "iterations" else pars[i]
                ax.set_title(f"sobel_feature(\n"
                             f"rgb_img,\n"
                             f"thresh_min={thresh_min}\n"
                             f"iterations={iterations})", size=fs, color=color)

    axes[0, 0].set_title("rgb_img", size=fs)
    axes[0, 1].set_title("Hue", size=fs)
    axes[0, 2].set_title("Lightness", size=fs, color="r")
    axes[0, 3].set_title("Saturation", size=fs, color="r")

    fig.suptitle(f"Sobel feature {tuning} tuning", size=fs * 2)
    fig.savefig(f"output_images/sobel_feature_{tuning}_tuning")


def plot_feature(rgb_sample):
    # Grid variables
    rows = len(rgb_sample)
    cols = 6
    y, x, c = rgb_sample[0].shape
    width = 8

    fig_size = (cols * width, rows * width * y / x * 0.9)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fs = round(fig_size[0]) / 1.5

    for row in range(rows):
        rgb_img = rgb_sample[row]

        ax = axes[row, 0]
        ax.imshow(rgb_img)

        ax = axes[row, 1]
        white_mask = white_feature(rgb_img)
        color_mask = np.dstack((white_mask, white_mask * 0, white_mask * 0))
        ax.imshow(color_mask)

        ax = axes[row, 2]
        yellow_mask = yellow_feature(rgb_img)
        color_mask = np.dstack((yellow_mask * 0, yellow_mask, yellow_mask * 0))
        ax.imshow(color_mask)

        ax = axes[row, 3]
        sobel_mask = sobel_feature(rgb_img)
        color_mask = np.dstack((sobel_mask * 0, sobel_mask * 0, sobel_mask))
        ax.imshow(color_mask)

        ax = axes[row, 4]
        red = white_mask
        green = yellow_mask
        blue = sobel_mask
        color_mask = np.dstack((red, green, blue))
        ax.imshow(color_mask)

        ax = axes[row, 5]
        final_mask = feature_mask(rgb_img)
        ax.imshow(final_mask, cmap="gray")

    axes[0, 0].set_title("rgb_img", size=fs)
    axes[0, 1].set_title("white feature", size=fs)
    axes[0, 2].set_title("yellow feature", size=fs)
    axes[0, 3].set_title("sobel feature", size=fs)
    axes[0, 4].set_title("composition", size=fs)
    axes[0, 5].set_title("feature mask", size=fs)
    fig.suptitle(f"Feature composistion", size=fs * 2)
    fig.savefig(f"output_images/feature_composition")


def plot_pipeline(rgb_sample):
    # Grid variables
    rows = len(rgb_sample)
    cols = 6
    y, x, c = rgb_sample[0].shape
    width = 8

    fig_size = (cols * width, rows * width * y / x)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fs = round(fig_size[0]) / 1.5

    for row in range(rows):
        # Undistorted rgb with roi
        rgb_img = rgb_sample[row]
        rgb_undist_img = undistort(rgb_img)
        roi_img = roi_image(rgb_undist_img)
        ax = axes[row, 0]
        ax.imshow(roi_img)

        # Binary feature mask
        feature_binary = feature_mask(rgb_undist_img)
        ax = axes[row, 1]
        ax.imshow(feature_binary, cmap="gray")

        # Perspective transform of rgb roi for visualization
        rgb_warped_img, trsf_mtx, trsf_mtx_inv = warp(
            rgb_undist_img, ROI_SRC, ROI_DST)
        ax = axes[row, 2]
        ax.imshow(rgb_warped_img)

        # Perspective transform of binary feature mask
        warped_feature_binary, _, _ = warp(
            feature_binary, roi_src=ROI_SRC, roi_dst=ROI_DST)
        # lane_mask = create_feature(rgb_warped_img)
        ax = axes[row, 3]
        ax.imshow(warped_feature_binary, cmap="gray")

        # Fit by sliding windows
        left_line, right_line, sliding_windows = fit_lines_by_windows(
            warped_feature_binary, Line(), Line())
        ax = axes[row, 4]
        ax.imshow(sliding_windows)

        # Fit by recent lines
        left_line, right_line, search_window = fit_lines_by_recent_lines(
            warped_feature_binary, left_line, right_line)
        ax = axes[row, 5]
        ax.imshow(search_window)

    axes[0, 0].set_title("Undistorted rgb\n"
                         "(with roi)", size=fs)
    axes[0, 1].set_title("Binary feature", size=fs)
    axes[0, 2].set_title("Warped roi\n"
                         "(just for visualization)", size=fs)
    axes[0, 3].set_title("Warped binary feature", size=fs)
    axes[0, 4].set_title("Fit by\n"
                         "sliding windows", size=fs)
    axes[0, 5].set_title("Fit by\n"
                         "recent lines", size=fs)
    fig.suptitle(f"Lane detection pipeline", size=fs * 2)
    fig.savefig(f"./output_images/lane_detection_pipeline")


if __name__ == "__main__":
    TEST_DIR = "test_images"
    OUTPUT_DIR = "output_images"
    images = [mpimg.imread(f"{TEST_DIR}/{img}") for img in os.listdir(TEST_DIR)]
    images = [undistort(img) for img in images]

    # Undistort one of the calibration images
    img = mpimg.imread("camera_cal/calibration1.jpg")
    undist_img = cv2.undistort(img, mtx, dist, dst=None, newCameraMatrix=mtx)
    mpimg.imsave(f"output_images/calibration1_undistorted.jpg", undist_img)
    plot_undistort_example(img, undist_img,
                           save="./output_images/camera_calibration_example")

    # Undistort one of the test images
    img = mpimg.imread("./test_images/test1.jpg")
    undist_img = cv2.undistort(img, mtx, dist, dst=None, newCameraMatrix=mtx)
    mpimg.imsave(f"output_images/test1_undistorted.jpg", undist_img)
    plot_undistort_example(img, undist_img,
                           save="./output_images/undistort_example")

    # Create channel plot of the test images
    plot_channels(images)

    # Create the white feature tuning plot
    plot_white_feature(images)

    # Create the yellow feature tuning plots
    plot_yellow_feature(images, tuning="h")
    plot_yellow_feature(images, tuning="s")
    plot_yellow_feature(images, tuning="v")

    # Create the yellow feature tuning plot
    plot_sobel_feature(images, tuning="thresh_min")
    plot_sobel_feature(images, tuning="iterations")

    # Create the feature composition images
    plot_feature(images)

    # Create perspective transformation plots
    plot_perspective_transform(images[0:-1:2], mtx, dist, ROI_SRC, ROI_DST)

    # Create the main pipeline plot
    plot_pipeline(images)

    # Process one of the test images
    img = mpimg.imread("./test_images/test2.jpg")
    processed_img = process(img)
    plot_process_example(processed_img, f"output_images/test2_processed.jpg")
