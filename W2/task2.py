import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from collections import OrderedDict
from itertools import islice

sys.path.append("W1")
import voc_evaluation
from aicity_reader import read_annotations, read_detections, group_by_frame
from bounding_box import BoundingBox

h, w, nc = 1080, 1920, 1

def adaptive_background_estimator(image, mean, std, alpha=3, rho=0.5):
    mask = abs(image - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    segmentation[mask] = 255

    # provar d'actualitzar el bg cada X frames
    mean = np.where(mask, mean, rho * image + (1-rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (image-mean)**2 + (1-rho) * std**2))

    return segmentation, mean, std

def static_background_estimator(image, mean, std, alpha=3):
    segmentation = np.zeros((1080, 1920))
    segmentation[abs(image - mean) >= alpha * (std + 2)] = 255
    return segmentation

def postprocess_after_segmentation(seg):
    kernel = np.ones((2, 2), np.uint8)
    # seg = cv2.erode(seg, kernel, iterations=1)
    # seg = cv2.dilate(seg, kernel, iterations=1)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel)
    return seg


def get_bboxes(seg, frame_id):
    # print(seg.shape)
    new_seg = cv2.cvtColor(seg.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    output = []

    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50: continue  # Discard small contours
        # print("cv2.contourArea(c)", cv2.contourArea(c))
        x, y, w, h = rect
        # TODO: Bounding box
        bboxes.append([x, y, x + w, y + h])

    for i, bbox in enumerate(bboxes):
        # output[i]={'box': bbox,'id': i,'frame':frame_id}

        # TODO: pillar bicis a gt
        output.append(BoundingBox(id=i, label='car', frame=frame_id, xtl=bbox[0],
                                  ytl=bbox[1], xbr=bbox[2], ybr=bbox[3]))

        # new_seg = cv2.rectangle(new_seg, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
        # cv2.putText(seg,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))

    # print("hola", new_seg.shape)
    # cv2.imshow("Show",new_seg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return output

def get_mean_std():
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

def train(vidcap, train_len, results_path, saveResults=False):
    count = 0

    mean, std = get_mean_std()

    print("Mean and std computed")

    if saveResults:
        cv2.imwrite(results_path + "mean_train.png", mean)
        cv2.imwrite(results_path + "std_train.png", std)

    return mean, std

def eval(vidcap, mean, std, params, saveResults=False):
    bboxes_byframe = []

    init_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_id = init_frame
    for t in tqdm(range(params['num_frames_eval'])):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # segmentation, mean, std = adaptive_background_estimator(frame, mean, std, params['alpha'], params['rho'])
        segmentation = static_background_estimator(frame, mean, std, params['alpha'])

        # segmentation = postprocess_after_segmentation(segmentation)
        bboxes = get_bboxes(segmentation, frame_id)
        bboxes_byframe = bboxes_byframe + bboxes
        if saveResults:
            cv2.imwrite(params['results_path'] + f"seg_{str(frame_id)}_pp_{str(params['alpha'])}.bmp", segmentation.astype(int))
        frame_id += 1

    gt = read_annotations(params['gt_path'], grouped=True, use_parked=False)
    gt_filtered = OrderedDict(list(islice(gt.items(), init_frame, frame_id)))  # only use the specified frames
    rec, prec, ap = voc_evaluation.voc_eval(bboxes_byframe, gt_filtered, ovthresh=0.5, use_confidence=False)
    print(rec, prec, ap)

    return


if __name__ == '__main__':
    params = {
        'video_path': "./data/vdo.avi",
        'gt_path': './data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml',
        'results_path': './W2/output/',
        'num_frames_eval': 3,
        'alpha': 3,
        'rho': 0.5
    }

    vidcap = cv2.VideoCapture(params['video_path'])
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # TODO: frame_width, height, nchannels

    print("Total frames: ", frame_count)

    train_len = int(0.25 * frame_count)
    test_len = frame_count - train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    # Train
    mean, std = train(vidcap, train_len, params['results_path'], saveResults=False)

    # Evaluate
    eval(vidcap, mean, std, params, saveResults=True)
