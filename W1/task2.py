import numpy as np
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_iou

if __name__ == '__main__':    
    
    gt_path = '/Data/ai_challenge_s03_c010-full_annotation.xml'
    det_path = '/Data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'
    
    gt = read_annotations(gt_path)
    det = read_detections(det_path)
    
    annotations_grouped = group_by_frame(gt)
    
    # read annotations
    class_recs = []
    npos = 0
    
    for frame_id, boxes in annotations_grouped.items():
        bbox = np.array([det.box for det in boxes])
        detection = [False] * len(boxes)
        npos += len(boxes)
        class_recs.append({"bbox": bbox, "det": detection})
    
    # read detections
    image_ids = [x.frame for x in det]
    BB = np.array([x.box for x in det]).reshape(-1, 4)
    
    confidence = np.array([float(x.confidence) for x in det])
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