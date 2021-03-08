import logging
import numpy as np
import os

'''
ADAPTED CODE FROM
https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py
'''
def voc_ap(rec, prec):
    """Compute VOC AP given precision and recall, using (always)
    the 11-point method .
    """
    # 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0
    
    return ap




def voc_iou(BBGT,bb):
    '''
    Compute IoU between groundtruth bounding box = BBGT
    and detected bounding box = bb
    '''
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
          + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
          - inters)
    overlaps = inters/uni
    return overlaps

def mean_iou(BBGT, bb):
    BBGT = np.array(BBGT).reshape(-1, 4)
    bb = np.array(bb).reshape(-1, 4)
    overlaps = voc_iou(BBGT, bb)
    
    return np.mean(np.max(overlaps, axis=1))

def voc_eval(detections, annotations, ovthresh=0.5,is_confidence=True):
    """rec, prec, ap = voc_eval(detections,
                                annotations
                                ovthresh)
    Top level function that does the PASCAL VOC -like evaluation.
    Detections
    Annotations: GROUPED BY FRAME
    ovthresh: Overlap threshold (default = 0.5)
     """
    # read annotations
    class_recs = []
    npos = 0

    for frame_id, boxes in annotations.items():
        bbox = np.array([det.box() for det in boxes])
        det = [False] * len(boxes)
        npos += len(boxes)
        class_recs.append({"bbox": bbox, "det": det}) 

   
 
    # read detections
    image_ids = [x.id for x in detections]
    BB = np.array([x.box() for x in detections]).reshape(-1, 4)

    if is_confidence:
        confidence = np.array([float(x.confidence) for x in detections])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    # go down detections (dets) and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = voc_iou(BBGT,bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap
