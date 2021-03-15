import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append("W1")
import voc_evaluation
import aicity_reader
import bounding_box

h, w, nc = 1080, 1920, 1

def adaptive_background_estimator(image, mean, std, alpha=3, rho=0.5):
    mask = abs(image - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    segmentation[mask] = 255

    mean = np.where(mask, mean, rho * image + (1-rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (image-mean)**2 + (1-rho) * std**2))

    return segmentation, mean, std

def adaptive_background_estimator_old(image, mean, std, alpha=3, rho=0.5):
    bg_mask = abs(image - mean) < alpha * (std + 2)
    fg_mask = np.logical_not(bg_mask)

    segmentation = np.zeros((h, w))
    segmentation[fg_mask] = 255

    mean[bg_mask] = rho * image[bg_mask] + (1-rho) * mean[bg_mask]
    variance = rho * (image[bg_mask] - mean[bg_mask])**2 + (1-rho) * std[bg_mask]**2
    std[bg_mask] = np.sqrt(variance[bg_mask].reshape(h,w))

    return segmentation, mean, std

def postprocess_after_segmentation(seg):
    kernel = np.ones((5, 5), np.uint8)
    seg = cv2.erode(seg, kernel, iterations=1)
    seg = cv2.dilate(seg, kernel, iterations=1)
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
        bboxes.append(
            [x, y, x + w, y + h])  # Great concern this is not the same format as the BoundingBox class but whopsies Pol

    for i, bbox in enumerate(bboxes):
        # output[i]={'box': bbox,'id': i,'frame':frame_id}
        output.append(bounding_box.BoundingBox(id=i, label='car', frame=frame_id, xtl=bbox[0], ytl=bbox[1], xbr=bbox[2],
                                               ybr=bbox[3]))

        # new_seg = cv2.rectangle(new_seg,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(seg,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))

    # print("hola", new_seg.shape)
    # cv2.imshow("Show",new_seg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return output


def train(vidcap, train_len, saveResults=False):
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
    std = np.sqrt(M2/count)

    print("Mean and std computed")

    if saveResults:
        cv2.imwrite("./W2/output/mean_train.png", mean)
        cv2.imwrite("./W2/output/std_train.png", std)

    return mean, std

def eval(vidcap, mean, std, params, saveResults=False):
    frame_num = train_len
    img_list_processed = []
    bboxes_byframe = []

    annotations = aicity_reader.read_annotations(params['gt_path'])
    annotations_grouped = aicity_reader.group_by_frame(annotations)

    for t in tqdm(range(params['num_frames_eval'])):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        segmentation, mean, std = adaptive_background_estimator(frame, mean, std, params['alpha'], params['rho'])
        # segmentation = postprocess_after_segmentation(segmentation)
        bboxes = get_bboxes(segmentation, frame_num)
        bboxes_byframe = bboxes_byframe + bboxes
        frame_num += 1
        if saveResults:
            cv2.imwrite(params['results_path'] + f"seg_{str(t)}_pp_{str(params['alpha'])}.bmp", segmentation.astype(int))

    rec, prec, ap = voc_evaluation.voc_eval(bboxes_byframe, annotations_grouped, ovthresh=0.5)
    print(rec, prec, ap)

    return


if __name__ == '__main__':
    params = {
        'video_path': "./data/vdo.avi",
        'gt_path': './data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml',
        'results_path': './W2/output/',
        'num_frames_eval': 10,
        'alpha': 3,
        'rho': 0.5
    }

    vidcap = cv2.VideoCapture(params['video_path'])
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total frames: ", frame_count)

    train_len = int(0.25 * frame_count)
    test_len = frame_count - train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    # Train
    mean, std = train(vidcap, train_len, saveResults=False)

    # Evaluate
    eval(vidcap, mean, std, params, saveResults=True)
