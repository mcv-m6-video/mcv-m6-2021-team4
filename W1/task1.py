import cv2
from aicity_reader import read_annotations, read_detections, group_by_frame
from noise_generator import add_noise
from voc_evaluation import voc_eval
from utils import draw_boxes


def task1_1(paths, show, noise_params):

    gt = read_annotations(paths['gt'], grouped=False, use_parked=True)
    det = read_detections(paths['det'], grouped=True)

    grouped_gt = group_by_frame(gt)

    # if we want to replicate results
    # np.random.seed(10)

    cap = cv2.VideoCapture(paths['video'])
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if noise_params['add']:
        noisy_gt = add_noise(gt, noise_params, num_frames)
        grouped_noisy_gt = group_by_frame(noisy_gt)

    for frame_id in range(num_frames):
        _, frame = cap.read()

        if show['gt']:
            frame = draw_boxes(frame, grouped_gt[frame_id], color='g')

        if show['det']:
            frame = draw_boxes(frame, det[frame_id], color='b', det=True)

        if show['noisy']:
            frame = draw_boxes(frame, grouped_noisy_gt[frame_id], color='r')

        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:  # press q to quit
            break

        frame_id += 1

    cv2.destroyAllWindows()

    return


def task1_2(paths, ap=0.5):
    gt = read_annotations(paths['gt'], grouped=True, use_parked=True)
    det = read_detections(paths['det'])

    rec, prec, ap = voc_eval(det, gt, ap, use_confidence=True)
    print(ap)

    return


def run():
    paths = {
        'gt': '../data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml',
        'det': '../data/AICity_data/train/S03/c010/det/det_yolo3.txt',
        'video': '../data/vdo.avi'
    }

    show = {
        'gt': True,
        'det': False,
        'noisy': True
    }

    noise_params = {
        'add': True,
        'drop': 0.2,
        'generate_close': 0.2,
        'generate_random': 0.3,
        'type': 'specific',  # options: 'specific', 'gaussian', None
        'std': 40,  # pixels
        'position': False,
        'size': True,
        'keep_ratio': True
    }

    task1_1(paths, show, noise_params)
    task1_2(paths, ap=0.5)