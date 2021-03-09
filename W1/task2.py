import matplotlib.pyplot as plt
import numpy as np
from evaluation_metrics import *




annotations_path = '/Data/ai_challenge_s03_c010-full_annotation.xml'
detections_path = '/Data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'

annotations = read_annotations(annotations_path)
detections = read_detections(detections_path)

annotations_grouped = group_by_frame(annotations)

# read annotations
class_recs = []
npos = 0

for frame_id, boxes in annotations_grouped.items():
    bbox = np.array([det.box for det in boxes])
    det = [False] * len(boxes)
    npos += len(boxes)
    class_recs.append({"bbox": bbox, "det": det}) 

# read detections
image_ids = [x.frame for x in detections]
BB = np.array([x.box for x in detections]).reshape(-1, 4)

confidence = np.array([float(x.confidence) for x in detections])
# sort by confidence
sorted_ind = np.argsort(-confidence)
BB = BB[sorted_ind, :]
image_ids = [image_ids[x] for x in sorted_ind]

# go down detections (dets) and mark TPs and FPs
nd = len(image_ids)
tp = np.zeros(nd)
fp = np.zeros(nd)

mean_iou = []
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
        iou=np.mean(np.max(overlaps))
        mean_iou.append(iou)
