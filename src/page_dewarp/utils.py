import cv2
import numpy as np
import imutils
from skimage.filters import sobel
from skimage.transform import rotate, resize, hough_line, hough_line_peaks
from skimage.util import invert
from scipy.stats import mode
from skimage.feature import canny

def rotate_small_angle(img, angle):
    rotated = imutils.rotate_bound(img, int(angle))
    return rotated

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    aspect = w/h

    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
        pad_vert = (sh-new_h)/2
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor]*3

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def draw_grid(img, grid_size=(100, 100), color=(0, 0, 255), thickness=1):

    h, w, _ = img.shape
    for x in range(0, w, grid_size[0]):
        cv2.line(img, (x, 0), (x, h), color, thickness)
    for y in range(0, h, grid_size[1]):
        cv2.line(img, (0, y), (w, y), color, thickness)
    return img 

def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)

# find the horizotal projection of the image and get the global 
def get_skew_angle(img):
    sobel_image = invert(sobel(img))
    predicted_angle = 0
    highest_hp = 0

    for index, angle in enumerate(range(-1, 1)):
        rotate_sobel = rotate(sobel_image, angle, cval=1)
        hp = horizontal_projections(rotate_sobel)
        median_hp = np.median(hp)
        if highest_hp < median_hp:
            predicted_angle = angle
            highest_hp = median_hp

    finetuning_range = np.arange(predicted_angle - 1.0, predicted_angle + 1.0, 0.1)
    for index, angle in enumerate(finetuning_range):
        hp = horizontal_projections(rotate(sobel_image, angle, cval=1))
        median_hp = np.median(hp)
        if highest_hp < median_hp:
            predicted_angle = angle
            highest_hp = median_hp
    
    print("PP predicted angle:", predicted_angle)
    return predicted_angle

def skew_angle_hough_transform(image):
    edges = canny(image)
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    accum, angles, dists = hough_line_peaks(h, theta, d)

    most_common_angle = mode(np.around(angles, decimals=2))[0]

    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    print(skew_angle)
    return skew_angle

def skew_angle_hough_transform_avarage(image):
    edges = canny(image)
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    accum, angles, dists = hough_line_peaks(h, theta, d)

    angles_deg = np.rad2deg(angles)

    horizontal_angles = angles_deg[(angles_deg > 80) & (angles_deg < 100)]

    if len(horizontal_angles) == 0:
        print("No horizontal lines detected.")
        return 0

    # mean_angle_deg = np.mean(angles_deg) #test
    mean_angle_deg = np.mean(horizontal_angles)

    mean_angle = np.deg2rad(mean_angle_deg)

    skew_angle = mean_angle - np.pi / 2
    skew_angle_deg = np.rad2deg(skew_angle)

    if abs(skew_angle_deg) > 2:
        skew_angle_deg = 0

    print("Skew angle:", skew_angle_deg)

    return skew_angle_deg