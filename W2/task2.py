import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("W1")
import voc_evaluation
from aicity_reader import read_annotations, read_detections, group_by_frame
from utils import draw_boxes
from bg_estimation import static_bg_est, adaptive_bg_est, postprocess_fg, fg_bboxes

h, w, nc = 1080, 1920, 1

bg_est_method = {
    'static': static_bg_est,
    'adaptive': adaptive_bg_est
}

def train(vidcap, train_len, results_path, saveResults=False):
    count = 0
    mean = np.zeros((h, w))
    M2 = np.zeros((h, w))

    # Compute mean and std
    for t in tqdm(range(train_len)):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        count += 1
        delta = frame - mean
        mean += delta / count
        delta2 = frame - mean
        M2 += delta * delta2

    mean = mean
    std = np.sqrt(M2 / count)

    print("Mean and std computed")

    if saveResults:
        cv2.imwrite(results_path + "mean_train.png", mean)
        cv2.imwrite(results_path + "std_train.png", std)

    return mean, std

def eval(vidcap, mean, std, params, saveResults=False):
    gt = read_annotations(params['gt_path'], grouped=True, use_parked=False)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    detections = []
    annotations = {}
    for t in tqdm(range(params['num_frames_eval'])):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        segmentation, mean, std = bg_est_method[params['bg_est']](frame, mean, std, params)
        segmentation = postprocess_fg(segmentation)

        if saveResults:
            cv2.imwrite(params['results_path'] + f"seg_{str(frame_id)}_pp_{str(params['alpha'])}.bmp", segmentation.astype(int))

        det_bboxes = fg_bboxes(segmentation, frame_id)
        detections += det_bboxes

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        annotations[frame_id] = gt_bboxes

        if params['show_boxes']:
            seg = cv2.cvtColor(segmentation.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            seg_boxes = draw_boxes(image=seg, boxes=det_bboxes, color='b', linewidth=3)
            seg_boxes = draw_boxes(image=seg_boxes, boxes=gt_bboxes, color='g', linewidth=3)

            cv2.imshow("Segmentation mask with detected boxes and gt", seg_boxes)
            cv2.waitKey()

        frame_id += 1

    rec, prec, ap = voc_evaluation.voc_eval(detections, annotations, ovthresh=0.5, use_confidence=False)
    print(rec, prec, ap)

    return

if __name__ == '__main__':
    params = {
        'video_path': "./data/vdo.avi",
        'gt_path': './data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml',
        'results_path': './W2/output/',
        'num_frames_eval': 3,
        'bg_est': 'static',
        'alpha': 3,
        'rho': 0.021,
        'show_boxes': True
    }

    vidcap = cv2.VideoCapture(params['video_path'])
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Total frames: ", frame_count)

    train_len = int(0.25 * frame_count)
    test_len = frame_count - train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    # Train
    mean, std = train(vidcap, train_len, params['results_path'], saveResults=False)

    # Evaluate
    eval(vidcap, mean, std, params, saveResults=False)
