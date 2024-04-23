from pathlib import Path

from datetime import datetime as dt
import time

import numpy as np
import cv2
from cv2 import INTER_AREA, imread, rectangle, LINE_AA, circle, line, namedWindow
from cv2 import resize as cv2_resize
from scipy.optimize import minimize
from PIL import Image

from debug_utils import debug_show
from dewarp import RemappedImage
from mask import Mask
# from optimise import optimise_params
from options import cfg
from projection import project_xy
from normalisation import norm2pix
from solve import get_default_params
from spans import assemble_spans, keypoints_from_samples, sample_spans
from simple_utils import fltp
from keypoints import make_keypoint_index, project_keypoints

def imgsize(img):
    height, width = img.shape[:2]
    return "{}x{}".format(width, height)


def get_page_dims(corners, rough_dims, params):
    dst_br = corners[2].flatten()
    dims = np.array(rough_dims)

    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten()) ** 2)

    # res = minimize(objective, dims, method="Powell")
    res = minimize(objective, dims, method="SLSQP")
    dims = res.x
    print("  got page dims", dims[0], "x", dims[1])
    return dims

def draw_correspondences(img, dstpoints, projpts):
    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)
    for pts, color in [(projpts, (255, 0, 0)), (dstpoints, (0, 0, 255))]:
        for point in pts:
            circle(display, fltp(point), 3, color, -1, LINE_AA)
    for point_a, point_b in zip(projpts, dstpoints):
        line(display, fltp(point_a), fltp(point_b), (255, 255, 255), 1, LINE_AA)
    return display

def resize_to_screen(cv2_img, copy=False):
    height, width = cv2_img.shape[:2]
    if height < 1000 and width < 1000:
        return cv2_img
    else:
        scl_x = float(width) / cfg.image_opts.SCREEN_MAX_W
        scl_y = float(height) / cfg.image_opts.SCREEN_MAX_H
        scl = int(np.ceil(max(scl_x, scl_y)))
        if scl > 1.0:
            inv_scl = 1.0 / scl
            img = cv2_resize(cv2_img, (0, 0), None, inv_scl, inv_scl, INTER_AREA)
        elif copy:
            img = cv2_img.copy()
        else:
            img = cv2_img
        return img

def calculate_page_extents(small):
    height, width = small.shape[:2]
    print(height, width)
    xmin = cfg.image_opts.PAGE_MARGIN_X
    ymin = cfg.image_opts.PAGE_MARGIN_Y
    xmax, ymax = (width - xmin), (height - ymin)
    pagemask = np.zeros((height, width), dtype=np.uint8)
    rectangle(pagemask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)
    page_outline = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    return pagemask, page_outline

def contour_info(small, pagemask, text=True):
    c_type = "text" if text else "line"
    mask = Mask('', small, pagemask, c_type)
    return mask.contours()

def iteratively_assemble_spans(stem, small, pagemask, contour_list):
    spans = assemble_spans(stem, small, pagemask, contour_list)
    # Retry if insufficient spans
    try:
        if len(spans) < 3:
            print(f"  detecting lines because only {len(spans)} text spans")
            contour_list = contour_info(small, pagemask, text=False)  # lines not text
            spans = attempt_reassemble_spans(stem, small, pagemask, contour_list, spans)
        return spans
    except Exception:
        pass

def attempt_reassemble_spans(stem, small, pagemask, contour_list, prev_spans):
    new_spans = assemble_spans(stem, small, pagemask, contour_list)
    return new_spans if len(new_spans) > len(prev_spans) else prev_spans

def threshold(stem, cv2_img, small, page_dims, params):
    remap = RemappedImage(stem, cv2_img, small, page_dims, params)
    # return remap.threshfile
    return remap.pil_image

def optimise_params(dstpoints, span_counts, params):
    
    skip_process = False

    keypoint_index = make_keypoint_index(span_counts)

    def fitting(pvec):
        res = minimize(objective, pvec, method="SLSQP")
        return res
    
    def objective(pvec, grad=None):
        ppts = project_keypoints(pvec, keypoint_index)
        # print('pvec shape: ', pvec.shape)
        # print('ppts shape: ', ppts.shape)
        # print('dstpoints shape: ', dstpoints.shape)
        return np.sum((dstpoints - ppts) ** 2)

    print('span_counts: ', span_counts)
    print("  initial objective is", objective(params))
    if objective(params) < 0.0008 or (objective(params) < 0.002 and 35 < len(span_counts) < 150):
        print("  skipping optimization because objective is already low")
        skip_process = True
        return params, skip_process

    print("  optimizing", len(params), "parameters...")
    start = dt.now()

    res = minimize(objective, params, method="SLSQP")

    end = dt.now()
    print(f"  optimization took {round((end - start).total_seconds(), 2)} sec.")
    print(f"  final objective is {round(res.fun, 5)}")
    params = res.x
    return params, skip_process

def four_point_transform(image, pts):

    pts = np.float32(pts)
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def main(imgfile):
        
    time_start = time.time()
    
    cv2_img = imread(imgfile)
    file_path = Path(imgfile).resolve()
    small = resize_to_screen(cv2_img)

    size, resized = imgsize(cv2_img), imgsize(small)
    print(f"Loaded {file_path.name} at {size=} --> {resized=}")
    if cfg.debug_lvl_opt.DEBUG_LEVEL >= 3:
        debug_show(file_path.stem, 0.0, "original", small)

    pagemask, page_outline = calculate_page_extents(small)

    contour_list = contour_info(small, pagemask, text=True)
    spans = iteratively_assemble_spans(file_path.stem, small, pagemask, contour_list)

    if len(spans) < 1:
        print(f"skipping {file_path.stem} because only {len(spans)} spans")
        color_converted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(color_converted)
        return pil_image
    else:
        span_points = sample_spans(small.shape, spans)
        n_pts = sum(map(len, span_points))
        print(f"  got {len(spans)} spans with {n_pts} points.")

        corners, ycoords, xcoords = keypoints_from_samples(
            file_path.stem, small, pagemask, page_outline, span_points
        )
        rough_dims, span_counts, params = get_default_params(corners, ycoords, xcoords)

        dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))

        params, skip_process = optimise_params(dstpoints, span_counts, params)

        print('skip_process: ', skip_process)
        if skip_process:
            color_converted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            pil_image=Image.fromarray(color_converted)
            return pil_image

        page_dims = get_page_dims(corners, rough_dims, params)
        if np.any(page_dims < 0):
            print("Got a negative page dimension! Falling back to rough estimate")
            page_dims = rough_dims
        threshfile = threshold(file_path.stem, cv2_img, small, page_dims, params)

        time_end = time.time()
        print(f"Total preprocess time: {time_end - time_start:.5f}s")
        
        return threshfile
