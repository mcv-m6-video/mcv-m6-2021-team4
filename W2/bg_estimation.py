import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("W1")
sys.path.append("W2")
from bounding_box import BoundingBox
import voc_evaluation
from aicity_reader import read_annotations, read_detections, group_by_frame
from utils import draw_boxes
from bg_postprocess import temporal_filter, postprocess_fg, discard_overlapping_bboxes

color_space = {
    'grayscale':[cv2.COLOR_BGR2GRAY,1],
    'RGB': [cv2.COLOR_BGR2RGB,3],
    'HSV': [cv2.COLOR_BGR2HSV,3],
    'LAB': [cv2.COLOR_BGR2LAB,3],
    'YUV': [cv2.COLOR_BGR2YUV,3],
    'YCrCb': [cv2.COLOR_BGR2YCrCb,3],
    'H': [cv2.COLOR_BGR2HSV,1],
    'L': [cv2.COLOR_BGR2LAB,1],
    'CbCr':[cv2.COLOR_BGR2YCrCb,2]
}



def static_bg_est(image, frame_size, mean, std, params):
    alpha = params['alpha']
    h, w = frame_size
    segmentation = np.zeros((h, w))
    mask = abs(image - mean) >= alpha * (std + 2)

    nc = color_space[params['color_space']][1]
    if nc == 1:
        segmentation[mask] = 255
    else:
        if params['voting'] == 'unanimous' or nc == 2:
            voting = (np.count_nonzero(mask, axis=2) / nc) >= 1
        elif params['voting'] == 'simple':
            voting = (np.count_nonzero(mask, axis=2) / (nc // 2 + 1)) >= 1
        else:
            raise ValueError('Voting method does not exist')

        segmentation[voting] = 255

    return segmentation, mean, std

def adaptive_bg_est(image, frame_size, mean, std, params):
    alpha = params['alpha']
    rho = params['rho']
    h, w = frame_size
    mask = abs(image - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    nc = color_space[params['color_space']][1]

    if nc == 1:
        segmentation[mask] = 255
    else:
        if params['voting'] == 'unanimous' or nc == 2:
            voting = (np.count_nonzero(mask, axis=2) / nc) >= 1
        elif params['voting'] == 'simple':
            voting = (np.count_nonzero(mask, axis=2) / (nc // 2 + 1)) >= 1
        else:
            raise ValueError('Voting method does not exist')

        segmentation[voting] = 255

    # provar d'actualitzar el bg cada X frames
    mean = np.where(mask, mean, rho * image + (1 - rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (image - mean) ** 2 + (1 - rho) * std ** 2))

    return segmentation, mean, std

<<<<<<< HEAD
=======
def postprocess_fg(seg):
    kernel = np.ones((2, 2), np.uint8)
    # seg = cv2.erode(seg, kernel, iterations=1)
    # seg = cv2.dilate(seg, kernel, iterations=1)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel)
    return seg

def intersection_over_areas(bboxA, bboxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bboxA.xtl, bboxB.xtl)
    yA = max(bboxA.ytl, bboxB.ytl)
    xB = min(bboxA.xbr, bboxB.xbr)
    yB = min(bboxA.ybr, bboxB.ybr)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return interArea/bboxA.area, interArea/bboxB.area

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


def fg_bboxes(seg, frame_id):
    bboxes = []
    mask = cv2.imread(r'data\AICity_data\train\S03\c010\roi.jpg').astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # print(mask.unique)
    # print(mask.shape)
    # print(type(mask))
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
>>>>>>> 4421e0328946e9b81264e27491aac81ebdcdbf86

def fg_bboxes(seg, frame_id, params):
    bboxes = []
    roi = cv2.imread(params['roi_path'], cv2.IMREAD_GRAYSCALE) / 255
    segmentation = seg * roi
    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    idx = 0
    for c in contours:
        rect = cv2.boundingRect(c)
<<<<<<< HEAD
        if rect[2] < 50 or rect[3] < 50 or rect[2]/rect[3] < 0.8:
            continue  # Discard small contours
=======
        if rect in mask[mask>240]
        if (rect[2] < 50 or rect[3] < 50 ) or ( rect[2] > 500 or rect[3] > 500 ) :
            continue  # Discard small contours and super big ones
>>>>>>> 4421e0328946e9b81264e27491aac81ebdcdbf86

       

        x, y, w, h = rect
        
        bboxes.append(BoundingBox(id=idx, label='car', frame=frame_id, xtl=x,
                                  ytl=y, xbr=x+w, ybr=y+h))
        idx += 1

    return discard_overlapping_bboxes(bboxes)

bg_est_method = {
    'static': static_bg_est,
    'adaptive': adaptive_bg_est
}

def eval(vidcap, frame_size, mean, std, params):
    gt = read_annotations(params['gt_path'], grouped=True, use_parked=False)
    init_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_id = init_frame
    detections = []
    annotations = {}
    for t in tqdm(range(params['num_frames_eval'])):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, color_space[params['color_space']][0])
        if params['color_space'] == 'H':
            H, S, V = np.split(frame, 3, axis=2)
            frame = np.squeeze(H)
        if params['color_space'] == 'L':
            L, A, B = np.split(frame, 3, axis=2)
            frame = np.squeeze(L)
        if params['color_space'] == 'CbCr':
            Y, Cb, Cr = np.split(frame, 3, axis=2)
            frame = np.dstack((Cb, Cr))
            
        segmentation, mean, std = bg_est_method[params['bg_est']](frame,frame_size, mean, std, params)
        roi = cv2.imread(params['roi_path'], cv2.IMREAD_GRAYSCALE) / 255
        segmentation = segmentation * roi
        segmentation = postprocess_fg(segmentation)

        if params['save_results'] and frame_id >= 1169 and frame_id < 1229 : # if frame_id >= 535 and frame_id < 550
            cv2.imwrite(params['results_path'] + f"seg_{str(frame_id)}_pp_{str(params['alpha'])}.bmp", segmentation.astype(int))

        det_bboxes = fg_bboxes(segmentation, frame_id, params)
        detections += det_bboxes

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        annotations[frame_id] = gt_bboxes

        if params['show_boxes']:
            seg = cv2.cvtColor(segmentation.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            seg_boxes = draw_boxes(image=seg, boxes=det_bboxes, color='r', linewidth=3)
            seg_boxes = draw_boxes(image=seg_boxes, boxes=gt_bboxes, color='g', linewidth=3)

            cv2.imshow("Segmentation mask with detected boxes and gt", seg_boxes)
            if cv2.waitKey() == 113:  # press q to quit
                break

        frame_id += 1

    detections = temporal_filter(group_by_frame(detections), init=init_frame, end=frame_id)
    rec, prec, ap = voc_evaluation.voc_eval(detections, annotations, ovthresh=0.5, use_confidence=False)

    return ap

def train(vidcap, frame_size, train_len, params):
    count = 0
    h, w = frame_size
    nc = color_space[params['color_space']][1]
    if nc == 1:
        mean = np.zeros((h, w))
        M2 = np.zeros((h, w))
    else:
        mean = np.zeros((h, w, nc))
        M2 = np.zeros((h, w, nc))

    # Compute mean and std
    for t in tqdm(range(train_len)):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, color_space[params['color_space']][0])
        if params['color_space'] == 'H':
            H,S,V = np.split(frame,3,axis=2)
            frame=np.squeeze(H)
        if params['color_space'] == 'L':
            L,A,B = np.split(frame,3,axis=2)
            frame=np.squeeze(L)
        if params['color_space'] == 'CbCr':
            Y,Cb,Cr= np.split(frame,3,axis=2)
            frame=np.dstack((Cb,Cr))
        count += 1
        delta = frame - mean
        mean += delta / count
        delta2 = frame - mean
        M2 += delta * delta2

    mean = mean
    std = np.sqrt(M2 / count)

    print("Mean and std computed")

    if params['save_results']:
        cv2.imwrite(params['results_path'] + "mean_train.png", mean)
        cv2.imwrite(params['results_path'] + "std_train.png", std)

    return mean, std

def train_sota(vidcap, train_len, backSub):
    print("Training SOTA")
    for t in tqdm(range(train_len)):
        #update the background model
        _ ,frame = vidcap.read()
        backSub.apply(frame)
    
    return backSub #return backSub updated

def eval_sota(vidcap, test_len, backSub, params):
    print("Evaluating SOTA")  
    gt = read_annotations(params["gt_path"], grouped=True, use_parked=False)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

    detections = []
    annotations = {}

    for t in tqdm(range(test_len)):

        _ ,frame = vidcap.read()

        segmentation = backSub.apply(frame)

        roi = cv2.imread(params['roi_path'], cv2.IMREAD_GRAYSCALE) / 255
        segmentation = segmentation * roi
        segmentation = postprocess_fg(segmentation)
        det_bboxes = fg_bboxes(segmentation, frame_id, params)
        detections += det_bboxes

        segmentation = cv2.cvtColor(segmentation.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        annotations[frame_id] = gt_bboxes

        text_bboxes = "nbb" #no bouning boxes
        if params['show_boxes']:
            segmentation = draw_boxes(image=segmentation, boxes=gt_bboxes, color='g', linewidth=3)
            cv2.rectangle(frame, (10, 2), (120,20), (255,255,255), -1)
            cv2.putText(frame, params["sota_method"]+" - "+str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            segmentation = draw_boxes(image=segmentation, boxes=det_bboxes, color='r', linewidth=3)
            cv2.imshow("Segmentation mask with detected boxes and gt", segmentation)
            cv2.imshow('Frame', frame)
            text_bboxes = ""


        if params['save_results'] and frame_id >= 1169 and frame_id < 1229 : # if frame_id >= 535 and frame_id < 550
            cv2.imwrite(params['results_path'] + f"seg_{str(frame_id)}_pp_{params['sota_method']}_{text_bboxes}.bmp", segmentation.astype(int))

            if cv2.waitKey() == 113:  # press q to quit
                break

        if params['save_results'] and frame_id >= 1169 and frame_id < 1229:

            segmentation = draw_boxes(image=segmentation, boxes=gt_bboxes, color='g', linewidth=3)
            segmentation = draw_boxes(image=segmentation, boxes=det_bboxes, color='r', linewidth=3)

            cv2.imwrite(f"seg/{str(frame_id)}.png", segmentation.astype(int))

        frame_id += 1

    detections = temporal_filter(group_by_frame(detections), init=535, end=frame_id)
    rec, prec, ap = voc_evaluation.voc_eval(detections, annotations, ovthresh=0.5, use_confidence=False)   
    return ap
