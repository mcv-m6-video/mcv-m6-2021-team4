import sys
from copy import deepcopy
import numpy as np
import cv2

sys.path.append("W1")
from bounding_box import BoundingBox, intersection_over_union, intersection_over_areas

def temporal_filter(detections, init, end):
    good_detections = []

    if init in detections:
        for d in detections[init]:
            good_detections.append(d)

    if end-1 in detections:
        for d in detections[end-1]:
            good_detections.append(d)

    iou_thr = 0.6
    for current_frame in range(init+1, end-1):
        if current_frame not in detections:
            continue
        det_current = detections[current_frame]
        det_prev = []
        det_next = []
        if current_frame-1 in detections:
            det_prev = detections[current_frame-1]
        if current_frame+1 in detections:
            det_next = detections[current_frame+1]

        for d_curr in det_current:
            max_iou_prev = 0
            max_iou_next = 0

            for d_prev in det_prev:
                iou_prev = intersection_over_union(d_curr, d_prev)
                max_iou_prev = max(max_iou_prev, iou_prev)

            for d_next in det_next:
                iou_next = intersection_over_union(d_curr, d_next)
                max_iou_next = max(max_iou_next, iou_next)

            if max_iou_prev >= iou_thr or max_iou_next >= iou_thr:
                good_detections.append(d_curr)

    return good_detections

def postprocess_fg(seg):

    kernel = np.ones((2, 2), np.uint8)
    seg = cv2.erode(seg, kernel, iterations=1)
    kernel = np.ones((3,4), np.uint8)
    seg = cv2.dilate(seg, kernel, iterations=1)

    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((7, 4), np.uint8))
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((4, 7), np.uint8))

    return seg

def discard_overlapping_bboxes(bboxes):
    ioa_thr = 0.7
    idxA = 0
    tmp_bboxes = deepcopy(bboxes)
    while idxA < len(bboxes)-1:
        discard = []
        discard_tmp = []
        discardA = False

        bboxA = bboxes[idxA]
        del tmp_bboxes[0]

        for idxB, bboxB in enumerate(tmp_bboxes):
            ioaA, ioaB = intersection_over_areas(bboxA, bboxB)

            if ioaA > ioa_thr or ioaB > ioa_thr:
                if ioaA > ioaB:
                    discardA = True
                    if idxA not in discard:
                        discard.append(idxA)
                else:
                    discard.append(idxA+idxB+1)
                    discard_tmp.append(idxB)

        discarded=0
        for d in sorted(discard):
            del bboxes[d-discarded]
            discarded += 1

        discarded_tmp = 0
        for d in sorted(discard_tmp):
            del tmp_bboxes[d-discarded_tmp]
            discarded_tmp += 1

        if not discardA:
            idxA +=1

        if len(tmp_bboxes) == 0:
            break

    return bboxes