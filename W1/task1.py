import xmltodict
import cv2
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict

img_shape = [1080, 1920]

colors = {
    'r': (0, 0, 255),
    'g': (0, 255, 0),
    'b': (255, 0, 0)
}


# import ffmpeg
# (
#     ffmpeg
#         .input('./data/AICity_data/train/S03/c010/vdo.avi')
#         .filter('fps', fps=1)
#         .output('test/%d.png',
#                 start_number=0)
#         .run(capture_stdout=True, capture_stderr=True)
#  )


class BoundingBox:

    def __init__(self, id, label, frame, xtl, ytl, xbr, ybr, occluded=None, parked=None, confidence=None):
        self.id = id
        self.label = label
        self.frame = frame
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.occluded = occluded
        self.parked = parked
        self.confidence = confidence

    @property
    def box(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def center(self):
        return [(self.xtl + self.xbr) // 2, (self.ytl + self.ybr) // 2]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ybr - self.ytl)

    def area(self):
        return self.width * self.height

    def shift_position(self, center):
        self.xtl = center[0] - self.width/2
        self.ytl = center[1] - self.height/2
        self.xbr = center[0] + self.width/2
        self.ybr = center[1] + self.height/2
        return

    def resize(self, size):

        # check if width and height > 0?

        h, w = size
        c = self.center
        self.xtl = c[0] - w/2
        self.ytl = c[1] - h/2
        self.xbr = c[0] + w/2
        self.ybr = c[1] + h/2
        return

    # tmp name
    def inside_image(self):
        h, w = img_shape

        if self.xtl < 0:
            self.xtl = 0

        if self.ytl < 0:
            self.ytl = 0

        if self.xbr >= w:
            self.xbr = w - 1

        if self.ybr >= h:
            self.ybr = h - 1

        return


def read_annotations(path):
    with open(path) as f:
        tracks = xmltodict.parse(f.read())['annotations']['track']

    annotations = []
    for track in tracks:
        id = track['@id']
        label = track['@label']

        if label != 'car':
            continue

        for box in track['box']:
            annotations.append(BoundingBox(
                id=int(id),
                label=label,
                frame=int(box['@frame']),
                xtl=float(box['@xtl']),
                ytl=float(box['@ytl']),
                xbr=float(box['@xbr']),
                ybr=float(box['@ybr']),
                occluded=box['@occluded'] == '1',
                parked=box['attribute']['#text'] == 'true'
            ))

    return annotations


def read_detections(path):
    """
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """

    with open(path) as f:
        lines = f.readlines()

    detections = []
    for line in lines:
        det = line.split(sep=',')
        detections.append(BoundingBox(
            id=int(det[1]),
            label='car',
            frame=int(det[0]) - 1,
            xtl=float(det[2]),
            ytl=float(det[3]),
            xbr=float(det[2]) + float(det[4]),
            ybr=float(det[3]) + float(det[5]),
            confidence=float(det[6])
        ))

    return detections


def group_by_frame(boxes):
    grouped = defaultdict(list)
    for box in boxes:
        grouped[box.frame].append(box)
    return OrderedDict(sorted(grouped.items()))


def draw_boxes(image, frame_id, boxes, color='g', det=False):
    rgb = colors[color]
    for box in boxes:
        if box.frame == frame_id:
            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), rgb, 2)
            if det:
                cv2.putText(image, str(box.confidence), (int(box.xtl), int(box.ytl) - 5), cv2.FONT_ITALIC, 0.6, rgb, 2)
    return image


def add_specific_noise(box, noise_params):
    if noise_params['position']:
        noisy_center = box.center + np.random.normal(0, noise_params['std'], 2)
        box.shift_position(noisy_center)

    if noise_params['size']:
        # generate this random width/height proportional to the initial width/height? (e.g. width * [0.95,1.05])
        if noise_params['keep_ratio']:
            size_noise = np.random.normal(0, noise_params['std'], 1)
            h = box.height + size_noise[0]
        else:
            size_noise = np.random.normal(0, noise_params['std'], 2)
            h = box.height + size_noise[1]

        w = box.width + size_noise[0]
        box.resize([h, w])

    return


def add_gaussian_noise(box, noise_params):
    noisy_coords = np.random.normal(0, noise_params['std'], 4)
    box.xtl = box.xtl + noisy_coords[0]
    box.ytl = box.ytl + noisy_coords[1]
    box.xbr = box.xbr + noisy_coords[2]
    box.ybr = box.ybr + noisy_coords[3]

    return


def add_noise(annotations, noise_params):
    noisy_annotations = []
    for box in annotations:

        # remove box
        if np.random.random() <= noise_params['drop']:
            continue

        # generate box
        if np.random.random() <= noise_params['generate']:
            new_box = deepcopy(box)
            noise['gaussian'](new_box, noise_params)  # add random noise to the new box (specify another std?)
            noisy_annotations.append(new_box)

        b = deepcopy(box)
        # add noise to existing box
        if noise_params['type'] is not None:
            noise[noise_params['type']](b, noise_params)

        noisy_annotations.append(b)

    return noisy_annotations


noise = {
    'gaussian': add_gaussian_noise,
    'specific': add_specific_noise
}

if __name__ == '__main__':
    annotations_path = '../../data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'
    detections_path = '../../data/AICity_data/train/S03/c010/det/det_yolo3.txt'
    video_path = '../../data/AICity_data/train/S03/c010/vdo.avi'

    # argparse ?

    show_annotations = True
    show_detections = False
    show_noisy = True

    noise_params = {
        'add': True,  # tmp name
        'drop': 0.0,
        'generate': 0.0,
        'type': 'specific',
        'std': 40,
        'position': False,
        'size': True,
        'keep_ratio': True
    }

    annotations = read_annotations(annotations_path)
    detections = read_detections(detections_path)

    grouped = group_by_frame(annotations)

    # for frame_id, boxes in grouped.items():
    #     for box in boxes:


    # is this necessary? we could compute the w and h randomly as well
    mean_w = 0
    mean_h = 0
    for b in annotations:
        mean_w += b.width
        mean_h += b.height
    mean_w /= len(annotations)
    mean_h /= len(annotations)

    # np.random.seed(10)

    if noise_params['add']:
        noisy_annotations = add_noise(annotations, noise_params)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prob_generate = 0.3

    while True:
        _, frame = cap.read()

        if noise_params['add'] and np.random.random() <= prob_generate:
            cx = np.random.randint(0, img_shape[1])
            cy = np.random.randint(0, img_shape[0])
            w = np.random.normal(mean_w, 50)
            h = np.random.normal(mean_h, 50)
            noisy_annotations.append(BoundingBox(
                id=-1,
                label='car',
                frame=frame_id,
                xtl=cx - w/2,
                ytl=cy - h/2,
                xbr=cx + w/2,
                ybr=cy + h/2
            ))

        if show_annotations:
            frame = draw_boxes(frame, frame_id, annotations, color='g')

        if show_detections:
            frame = draw_boxes(frame, frame_id, detections, color='b', det=True)

        if show_noisy:
            frame = draw_boxes(frame, frame_id, noisy_annotations, color='r')

        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:  # press q to quit
            break

        frame_id += 1













# # -*- coding: utf-8 -*-
# # Copyright (c) Facebook, Inc. and its affiliates.
#
# import logging
# import numpy as np
# import os
# import tempfile
# import xml.etree.ElementTree as ET
# from collections import OrderedDict, defaultdict
# from functools import lru_cache
# import torch
#
# from detectron2.data import MetadataCatalog
# from detectron2.utils import comm
# from detectron2.utils.file_io import PathManager
#
# from .evaluator import DatasetEvaluator
#
#
# class PascalVOCDetectionEvaluator(DatasetEvaluator):
#     """
#     Evaluate Pascal VOC style AP for Pascal VOC dataset.
#     It contains a synchronization, therefore has to be called from all ranks.
#
#     Note that the concept of AP can be implemented in different ways and may not
#     produce identical results. This class mimics the implementation of the official
#     Pascal VOC Matlab API, and should produce similar but not identical results to the
#     official API.
#     """
#
#     def __init__(self, dataset_name):
#         """
#         Args:
#             dataset_name (str): name of the dataset, e.g., "voc_2007_test"
#         """
#         self._dataset_name = dataset_name
#         meta = MetadataCatalog.get(dataset_name)
#
#         # Too many tiny files, download all to local for speed.
#         annotation_dir_local = PathManager.get_local_path(
#             os.path.join(meta.dirname, "Annotations/")
#         )
#         self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
#         self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
#         self._class_names = meta.thing_classes
#         assert meta.year in [2007, 2012], meta.year
#         self._is_2007 = meta.year == 2007
#         self._cpu_device = torch.device("cpu")
#         self._logger = logging.getLogger(__name__)
#
#     def reset(self):
#         self._predictions = defaultdict(list)  # class name -> list of prediction strings
#
#     def process(self, inputs, outputs):
#         for input, output in zip(inputs, outputs):
#             image_id = input["image_id"]
#             instances = output["instances"].to(self._cpu_device)
#             boxes = instances.pred_boxes.tensor.numpy()
#             scores = instances.scores.tolist()
#             classes = instances.pred_classes.tolist()
#             for box, score, cls in zip(boxes, scores, classes):
#                 xmin, ymin, xmax, ymax = box
#                 # The inverse of data loading logic in `datasets/pascal_voc.py`
#                 xmin += 1
#                 ymin += 1
#                 self._predictions[cls].append(
#                     f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
#                 )
#
#     def evaluate(self):
#         """
#         Returns:
#             dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
#         """
#         all_predictions = comm.gather(self._predictions, dst=0)
#         if not comm.is_main_process():
#             return
#         predictions = defaultdict(list)
#         for predictions_per_rank in all_predictions:
#             for clsid, lines in predictions_per_rank.items():
#                 predictions[clsid].extend(lines)
#         del all_predictions
#
#         self._logger.info(
#             "Evaluating {} using {} metric. "
#             "Note that results do not use the official Matlab API.".format(
#                 self._dataset_name, 2007 if self._is_2007 else 2012
#             )
#         )
#
#         with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
#             res_file_template = os.path.join(dirname, "{}.txt")
#
#             aps = defaultdict(list)  # iou -> ap per class
#             for cls_id, cls_name in enumerate(self._class_names):
#                 lines = predictions.get(cls_id, [""])
#
#                 with open(res_file_template.format(cls_name), "w") as f:
#                     f.write("\n".join(lines))
#
#                 for thresh in range(50, 100, 5):
#                     rec, prec, ap = voc_eval(
#                         res_file_template,
#                         self._anno_file_template,
#                         self._image_set_path,
#                         cls_name,
#                         ovthresh=thresh / 100.0,
#                         use_07_metric=self._is_2007,
#                     )
#                     aps[thresh].append(ap * 100)
#
#         ret = OrderedDict()
#         mAP = {iou: np.mean(x) for iou, x in aps.items()}
#         ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
#         return ret
#
#
# ##############################################################################
# #
# # Below code is modified from
# # https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# # --------------------------------------------------------
# # Fast/er R-CNN
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Bharath Hariharan
# # --------------------------------------------------------
#
# """Python implementation of the PASCAL VOC devkit's AP evaluation code."""
#
#
# @lru_cache(maxsize=None)
# def parse_rec(filename):
#     """Parse a PASCAL VOC xml file."""
#     with PathManager.open(filename) as f:
#         tree = ET.parse(f)
#     objects = []
#     for obj in tree.findall("object"):
#         obj_struct = {}
#         obj_struct["name"] = obj.find("name").text
#         obj_struct["pose"] = obj.find("pose").text
#         obj_struct["truncated"] = int(obj.find("truncated").text)
#         obj_struct["difficult"] = int(obj.find("difficult").text)
#         bbox = obj.find("bndbox")
#         obj_struct["bbox"] = [
#             int(bbox.find("xmin").text),
#             int(bbox.find("ymin").text),
#             int(bbox.find("xmax").text),
#             int(bbox.find("ymax").text),
#         ]
#         objects.append(obj_struct)
#
#     return objects
#
#
# def voc_ap(rec, prec, use_07_metric=False):
#     """Compute VOC AP given precision and recall. If use_07_metric is true, uses
#     the VOC 07 11-point method (default:False).
#     """
#     if use_07_metric:
#         # 11 point metric
#         ap = 0.0
#         for t in np.arange(0.0, 1.1, 0.1):
#             if np.sum(rec >= t) == 0:
#                 p = 0
#             else:
#                 p = np.max(prec[rec >= t])
#             ap = ap + p / 11.0
#     else:
#         # correct AP calculation
#         # first append sentinel values at the end
#         mrec = np.concatenate(([0.0], rec, [1.0]))
#         mpre = np.concatenate(([0.0], prec, [0.0]))
#
#         # compute the precision envelope
#         for i in range(mpre.size - 1, 0, -1):
#             mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
#
#         # to calculate area under PR curve, look for points
#         # where X axis (recall) changes value
#         i = np.where(mrec[1:] != mrec[:-1])[0]
#
#         # and sum (\Delta recall) * prec
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap
#
#
# def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
#     """rec, prec, ap = voc_eval(detpath,
#                                 annopath,
#                                 imagesetfile,
#                                 classname,
#                                 [ovthresh],
#                                 [use_07_metric])
#
#     Top level function that does the PASCAL VOC evaluation.
#
#     detpath: Path to detections
#         detpath.format(classname) should produce the detection results file.
#     annopath: Path to annotations
#         annopath.format(imagename) should be the xml annotations file.
#     imagesetfile: Text file containing the list of images, one image per line.
#     classname: Category name (duh)
#     [ovthresh]: Overlap threshold (default = 0.5)
#     [use_07_metric]: Whether to use VOC07's 11 point AP computation
#         (default False)
#     """
#     # assumes detections are in detpath.format(classname)
#     # assumes annotations are in annopath.format(imagename)
#     # assumes imagesetfile is a text file with each line an image name
#
#     # first load gt
#     # read list of images
#     with PathManager.open(imagesetfile, "r") as f:
#         lines = f.readlines()
#     imagenames = [x.strip() for x in lines]
#
#     # load annots
#     recs = {}
#     for imagename in imagenames:
#         recs[imagename] = parse_rec(annopath.format(imagename))
#
#     # extract gt objects for this class
#     class_recs = {}
#     npos = 0
#     for imagename in imagenames:
#         R = [obj for obj in recs[imagename] if obj["name"] == classname]
#         bbox = np.array([x["bbox"] for x in R])
#         difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
#         # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
#         det = [False] * len(R)
#         npos = npos + sum(~difficult)
#         class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
#
#     # read dets
#     detfile = detpath.format(classname)
#     with open(detfile, "r") as f:
#         lines = f.readlines()
#
#     splitlines = [x.strip().split(" ") for x in lines]
#     image_ids = [x[0] for x in splitlines]
#     confidence = np.array([float(x[1]) for x in splitlines])
#     BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
#
#     # sort by confidence
#     sorted_ind = np.argsort(-confidence)
#     BB = BB[sorted_ind, :]
#     image_ids = [image_ids[x] for x in sorted_ind]
#
#     # go down dets and mark TPs and FPs
#     nd = len(image_ids)
#     tp = np.zeros(nd)
#     fp = np.zeros(nd)
#     for d in range(nd):
#         R = class_recs[image_ids[d]]
#         bb = BB[d, :].astype(float)
#         ovmax = -np.inf
#         BBGT = R["bbox"].astype(float)
#
#         if BBGT.size > 0:
#             # compute overlaps
#             # intersection
#             ixmin = np.maximum(BBGT[:, 0], bb[0])
#             iymin = np.maximum(BBGT[:, 1], bb[1])
#             ixmax = np.minimum(BBGT[:, 2], bb[2])
#             iymax = np.minimum(BBGT[:, 3], bb[3])
#             iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
#             ih = np.maximum(iymax - iymin + 1.0, 0.0)
#             inters = iw * ih
#
#             # union
#             uni = (
#                 (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
#                 + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
#                 - inters
#             )
#
#             overlaps = inters / uni
#             ovmax = np.max(overlaps)
#             jmax = np.argmax(overlaps)
#
#         if ovmax > ovthresh:
#             if not R["difficult"][jmax]:
#                 if not R["det"][jmax]:
#                     tp[d] = 1.0
#                     R["det"][jmax] = 1
#                 else:
#                     fp[d] = 1.0
#         else:
#             fp[d] = 1.0
#
#     # compute precision recall
#     fp = np.cumsum(fp)
#     tp = np.cumsum(tp)
#     rec = tp / float(npos)
#     # avoid divide by zero in case the first detection matches a difficult
#     # ground truth
#     prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#     ap = voc_ap(rec, prec, use_07_metric)
#
#     return rec, prec, ap
