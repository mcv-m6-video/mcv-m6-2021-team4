import cv2
import numpy as np
from bounding_box import BoundingBox
from aicity_reader import read_annotations, read_detections, group_by_frame
from noise_generator import add_noise
from voc_evaluation import voc_eval
from utils import draw_boxes

img_shape = [1080, 1920]


def task1_1(gt_path, det_path):
    video_path = '../../data/AICity_data/train/S03/c010/vdo.avi'

    show_gt = True
    show_det = False
    show_noisy = True

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

    gt = read_annotations(gt_path)
    det = read_detections(det_path)

    grouped_gt = group_by_frame(gt)
    grouped_det = group_by_frame(det)

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

    for frame_id in range(num_frames):
        _, frame = cap.read()

        # generate a random BB in this frame
        if noise_params['add'] and np.random.random() <= noise_params['generate_random']:
            # center of the BB (cx, cy), width and height (w, h)
            cx = np.random.randint(mean_w//2, img_shape[1]-mean_w//2)
            cy = np.random.randint(mean_h//2, img_shape[0]-mean_h//2)
            w = np.random.normal(mean_w, 20)
            h = np.random.normal(mean_h, 20)
            noisy_gt.append(BoundingBox(
                id=-1,
                label='car',
                frame=frame_id,
                xtl=cx - w/2,
                ytl=cy - h/2,
                xbr=cx + w/2,
                ybr=cy + h/2
            ))

        if show_gt:
            frame = draw_boxes(frame, frame_id, grouped_gt[frame_id], color='g')

        if show_det:
            frame = draw_boxes(frame, frame_id, grouped_det[frame_id], color='b', det=True)

        if show_noisy:
            frame = draw_boxes(frame, frame_id, grouped_noisy_gt[frame_id], color='r')

        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:  # press q to quit
            break

        frame_id += 1

    cv2.destroyAllWindows()

    return


def task1_2(gt_path, det_path, ap=0.5):
    gt = read_annotations(gt_path)
    det = read_detections(det_path)

    grouped_gt = group_by_frame(gt)

    rec, prec, ap = voc_eval(det, grouped_gt, ap, is_confidence=True)
    print(ap)

    return


if __name__ == '__main__':
    gt_path = '../../data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'
    det_path = '../../data/AICity_data/train/S03/c010/det/det_yolo3.txt'

    task1_1(gt_path, det_path)
    task1_2(gt_path, det_path, ap=0.5)