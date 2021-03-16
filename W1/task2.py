import cv2
import matplotlib.pyplot as plt
from noise_generator import add_noise
from utils import draw_boxes, save_gif
import numpy as np
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_iou
import imageio
import os


def task2(gt_path, det_path, video_path, results_path):
    plot_frames_path = os.path.join(results_path, 'plot_frames/')
    video_frames_path = os.path.join(results_path, 'video_frames/')

    print(plot_frames_path)

    # If folder doesn't exist -> create it
    os.makedirs(plot_frames_path, exist_ok=True)
    os.makedirs(video_frames_path, exist_ok=True)

    show_det = True
    show_noisy = False

    gt = read_annotations(gt_path, grouped=False, use_parked=True)
    det = read_detections(det_path, grouped=True)

    grouped_gt = group_by_frame(gt)

    noise_params = {
        'add': False,
        'drop': 0.0,
        'generate_close': 0.0,
        'generate_random': 0.0,
        'type': 'specific',  # options: 'specific', 'gaussian', None
        'std': 40,  # pixels
        'position': False,
        'size': True,
        'keep_ratio': True
    }

    if noise_params['add']:
        noisy_gt = add_noise(gt, noise_params)
        grouped_noisy_gt = group_by_frame(noisy_gt)

    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    iou_list = {}

    for frame_id in range(20):
        _, frame = cap.read()

        frame = draw_boxes(frame, grouped_gt[frame_id], color='g')

        if show_det:
            frame = draw_boxes(frame, det[frame_id], color='b', det=True)
            frame_iou = mean_iou(det[frame_id], grouped_gt[frame_id], sort=True)

        if show_noisy:
            frame = draw_boxes(frame, grouped_noisy_gt[frame_id], color='r')
            frame_iou = mean_iou(grouped_noisy_gt[frame_id], grouped_gt[frame_id])

        iou_list[frame_id] = frame_iou

        plot = plot_iou(iou_list, num_frames)

        '''
        if show:
            fig.show()
            cv2.imshow('frame', frame)
            if cv2.waitKey() == 113:  # press q to quit
                break
        '''
        imageio.imwrite(video_frames_path + '{}.png'.format(frame_id), frame)
        plot.savefig(plot_frames_path + 'iou_{}.png'.format(frame_id))
        plt.close(plot)

        frame_id += 1

    save_gif(plot_frames_path, results_path + 'iou.gif')
    save_gif(video_frames_path, results_path + 'bbox.gif')
    # cv2.destroyAllWindows()

    return

def plot_iou(dict_iou, xmax):
    lists = sorted(dict_iou.items())  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y)
    ax.grid()
    ax.set(xlabel='frame', ylabel='IoU',
           title='IoU vs Time')

    # Used to keep the limits constant
    ax.set_ylim(0, 1)
    ax.set_xlim(0, xmax)

    return fig


def mean_iou(det, gt, sort=False):
    '''
    det: detections of one frame
    gt: annotations of one frame
    sort: False if we use modified GT,
          True if we have a confidence value for the detection
    '''
    if sort:
        BB = sort_by_confidence(det)
    else:
        BB = np.array([x.box for x in det]).reshape(-1, 4)

    BBGT = np.array([anot.box for anot in gt])

    nd = len(BB)
    mean_iou = []
    for d in range(nd):
        bb = BB[d, :].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = voc_iou(BBGT, bb)
            ovmax = np.max(overlaps)
            mean_iou.append(ovmax)

    return np.mean(mean_iou)


def sort_by_confidence(det):
    BB = np.array([x.box for x in det]).reshape(-1, 4)
    confidence = np.array([float(x.confidence) for x in det])
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    return BB


def run():
    gt_path = '../data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'
    det_path = '../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'
    video_path = '../data/vdo.avi'

    task2(gt_path, det_path, video_path=video_path, results_path='./results')