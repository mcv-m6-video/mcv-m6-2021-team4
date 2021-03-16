import numpy as np
import cv2
import sys
sys.path.append("W1")
from bounding_box import BoundingBox

h, w, nc = 1080, 1920, 1

def static_bg_est(image, mean, std, params):
    alpha = params['alpha']
    segmentation = np.zeros((1080, 1920))
    segmentation[abs(image - mean) >= alpha * (std + 2)] = 255
    return segmentation, mean, std

def adaptive_bg_est(image, mean, std, params):
    alpha = params['alpha']
    rho = params['rho']

    mask = abs(image - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    segmentation[mask] = 255

    # provar d'actualitzar el bg cada X frames
    mean = np.where(mask, mean, rho * image + (1-rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (image-mean)**2 + (1-rho) * std**2))

    return segmentation, mean, std

def postprocess_fg(seg):
    kernel = np.ones((2, 2), np.uint8)
    # seg = cv2.erode(seg, kernel, iterations=1)
    # seg = cv2.dilate(seg, kernel, iterations=1)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel)
    return seg

def fg_bboxes(seg, frame_id):
    bboxes = []
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50:
            continue  # Discard small contours

        x, y, w, h = rect

        # TODO: pillar bicis a gt
        bboxes.append(BoundingBox(id=idx, label='car', frame=frame_id, xtl=x,
                                  ytl=y, xbr=x+w, ybr=y+h))
        idx += 1

    return bboxes