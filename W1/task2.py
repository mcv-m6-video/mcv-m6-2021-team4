import cv2
import numpy as np
import matplotlib.pyplot as plt
from noise_generator import add_noise
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_iou
from utils import draw_boxes


def task2(gt_path, det_path, video_path):
    #video_path = '../../data/AICity_data/train/S03/c010/vdo.avi'

    show_det = False
    show_noisy = True

    gt = read_annotations(gt_path)
    det = read_detections(det_path)

    grouped_gt = group_by_frame(gt)
    grouped_det = group_by_frame(det)

    noise_params = {
        'add': True,
        'drop': 0.0,
        'generate_close': 0.0,
        'generate_random': 0.0,
        'type': 'specific',  # options: 'specific', 'gaussian', None
        'std': 40,  # pixels
        'position': False,
        'size': True,
        'keep_ratio': True
    }

    # to generate a BB randomly in an image, we use the mean width and
    # height of the annotated BBs so that they have similar statistics
    if noise_params['generate_random'] > 0.0:
        mean_w = 0
        mean_h = 0
        for b in gt:
            mean_w += b.width
            mean_h += b.height
        mean_w /= len(gt)
        mean_h /= len(gt)

    # if we want to replicate results
    # np.random.seed(10)

    if noise_params['add']:
        noisy_gt = add_noise(gt, noise_params)
        grouped_noisy_gt = group_by_frame(noisy_gt)

    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



    frame_list = []
    iou_plots = []
    iou_list = {}

    for frame_id in range(num_frames):
        _, frame = cap.read()

        frame = draw_boxes(frame, frame_id, grouped_gt[frame_id], color='g')

        if show_det:
            frame = draw_boxes(frame, frame_id, grouped_det[frame_id], color='b', det=True)
            frame_iou = mean_iou(grouped_det[frame_id],grouped_gt[frame_id],sort=True)

        if show_noisy:
            frame = draw_boxes(frame, frame_id, grouped_noisy_gt[frame_id], color='r')
            frame_iou = mean_iou(grouped_noisy_gt[frame_id],grouped_gt[frame_id])

        iou_list[frame_id] = frame_iou


        '''
        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:  # press q to quit
            break
        '''

        frame_list.append(frame)


        frame_id += 1

    cv2.destroyAllWindows()

    return

def plot_iou(dict_iou):
    lists = sorted(dict_iou.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.ylim(0,1)
    plt.xlim(0,len(lists))

    plt.show()


# IAN: S'HA CANVIAT L'ORDRE!!!!
def iou_frame(gt, det, sort=False):
    """
    det: detections of one frame
    gt: annotations of one frame
    sort: False if we use modified GT,
          True if we have a confidence value for the detection
    """
    if sort:
        BB = sort_by_confidence(det)
    else:
        BB = np.array([x.box for x in det]).reshape(-1, 4)

    BBGT = np.array([anot.box for anot in gt])

    nd = len(BB)
    iou_boxes = []
    for d in range(nd):
        bb = BB[d, :].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = voc_iou(BBGT, bb)
            ovmax = np.max(overlaps)
            iou_boxes.append(ovmax)

    return iou_boxes


# grouped in frames!!
def mean_iou(gt, det, detections=False):
    acc_iou = 0
    total_boxes = 0
    for frame_id, det_boxes in det.items():
        acc_iou += np.sum(iou_frame(gt[frame_id], det_boxes, detections))
        total_boxes += len(det_boxes)

    return acc_iou / total_boxes


# a on fiquem això?
def sort_by_confidence(det):
    BB = np.array([x.box for x in det]).reshape(-1, 4)
    confidence = np.array([float(x.confidence) for x in det])
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    return BB


if __name__ == '__main__':

    gt_path = '../../data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'
    det_path = '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'

    gt = read_annotations(gt_path)
    det = read_detections(det_path)

    gt_grouped = group_by_frame(gt)
    det_grouped = group_by_frame(det)

    noise_params = {
        'add': True,
        'drop': 0.4,
        'generate_close': 0.0,
        'generate_random': 0.0,
        'type': 'specific',  # options: 'specific', 'gaussian', None
        'std': 40,  # pixels
        'position': False,
        'size': True,
        'keep_ratio': True
    }

    cap = cv2.VideoCapture('../../data/AICity_data/train/S03/c010/vdo.avi')
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    noisy_gt = add_noise(gt, noise_params, num_frames)
    grouped_noisy_gt = group_by_frame(noisy_gt)

    mean_iou = mean_iou(gt_grouped, grouped_noisy_gt, detections=False)
    print(mean_iou)

