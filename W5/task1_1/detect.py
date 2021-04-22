import os, sys, cv2, argparse, torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

sys.path.append('./W1')
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_eval
from utils import draw_boxes

sys.path.append("./W2")
from bg_estimation import train_sota, eval_sota


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='faster',
                        choices=['faster', 'mog'])

    parser.add_argument('--seq', type=str, default='S03/c010',
                        help='sequence/camera from AICity dataset')

    parser.add_argument('--data_path', type=str, default="./data/AICity_data/train",
                        help='path to sequences of AICity')

    parser.add_argument('--results_dir', type=str, default="./W5/task1_1/detections",
                        help='path to save results')

    parser.add_argument('--show_boxes', action='store_true',
                        help='show bounding boxes')

    parser.add_argument('--save_results', action='store_true',
                        help='save detections')

    return parser.parse_args(args)


def args_to_params(args):
    return {
        'video_path': os.path.join(args.data_path, args.seq, 'vdo.avi'),
        'roi_path': os.path.join(args.data_path, args.seq, 'roi.jpg'),
        # 'gt_path': os.path.join(args.data_path, args.seq, 'ai_challenge_s03_c010-full_annotation.xml'),
        'gt_path': os.path.join(args.data_path, args.seq, 'gt/gt.txt'),
        'show_boxes': args.show_boxes,
        'sota_method': args.method,
        'save_results': args.save_results,
        'results_path': os.path.join(args.results_dir, args.method, args.seq)
    }

def save_detections(detections, det_path, num_frames):
    for frame_id in range(num_frames):
        if frame_id not in detections:
            continue

        for box in detections[frame_id]:
            # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                det = str(frame_id + 1) + ',-1,' + str(box.xtl) + ',' + str(box.ytl) + ',' + str(
                    box.width) + ',' + str(box.height) + ',' + str(1) + ',-1,-1,-1\n'

                with open(det_path, 'a') as f:
                    f.write(det)
    return


def fill_gt(gt, num_frames):
    filled_gt = {}
    for frame_id in range(num_frames):
        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        filled_gt[frame_id] = gt_bboxes
    return filled_gt


if __name__ == '__main__':
    args = parse_args()
    print('[INFO] Detecting boxes using ', args.method, ' method')

    params = args_to_params(args)

    os.makedirs(params['results_path'], exist_ok=True)
    det_path = os.path.join(params['results_path'], 'detections.txt')
    if os.path.exists(det_path):
        os.remove(det_path)

    if args.method == 'mog':
        vidcap = cv2.VideoCapture(params['video_path'])
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", frame_count)

        train_len = int(0.25 * frame_count)
        print('Train frames: ', train_len)

        backSub = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=2, backgroundRatio=0.7)
        backSub = train_sota(vidcap, train_len, backSub)

        vidcap = cv2.VideoCapture(params['video_path'])
        test_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Test frames: ', test_len)

        ap, detections = eval_sota(vidcap, test_len, backSub, params, init_frame=0, return_detections=True)
        print('AP: ', ap)

        save_detections(detections, det_path, frame_count)

        # gt = read_annotations(params['gt_path'], grouped=True, use_parked=True)
        gt = read_detections(os.path.join(args.data_path, args.seq, 'gt/gt.txt'), grouped=True)
        gt = fill_gt(gt, frame_count)
        det = read_detections(det_path, grouped=False)

        rec, prec, ap = voc_eval(det, gt, 0.5, use_confidence=True)
        print('AP from loaded detections: ', ap)

    else:
        model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
        print('[INFO] Using model: ', model)

        weights_path = './W3/results/task1_2_all/faster_rcnn/lr_0_001_iter_5000_batch_512'

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = os.path.join(weights_path, "model_final.pth")  # path to the model we just trained

        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        predictor = DefaultPredictor(cfg)

        vidcap = cv2.VideoCapture(params['video_path'])
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_id in tqdm(range(num_frames), desc='Detecting cars in video sequence...'):
            _, frame = vidcap.read()

            outputs = predictor(frame)

            pred_boxes = outputs["instances"].pred_boxes.to("cpu")
            scores = outputs["instances"].scores.to("cpu")

            for idx in range(len(pred_boxes)):
                box = pred_boxes[idx].tensor.numpy()[0]

                # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                det = str(frame_id + 1) + ',-1,' + str(box[0]) + ',' + str(box[1]) + ',' + str(
                    box[2] - box[0]) + ',' + str(box[3] - box[1]) + ',' + str(scores[idx].item()) + ',-1,-1,-1\n'

                with open(det_path, 'a+') as f:
                    f.write(det)

        # gt = read_annotations(os.path.join(args.data_path, args.seq, 'ai_challenge_s03_c010-full_annotation.xml'), grouped=True, use_parked=True)
        gt = read_detections(params['gt_path'], grouped=True)
        gt = fill_gt(gt, num_frames)
        det = read_detections(det_path, grouped=False)

        rec, prec, ap = voc_eval(det, gt, 0.5, use_confidence=True)
        print('AP: ', ap)


        if args.show_boxes:
            gt = read_annotations(params['gt_path'], grouped=True, use_parked=True)
            det = read_detections(det_path, grouped=True)

            vidcap = cv2.VideoCapture(params['video_path'])
            # vidcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)  # to start from frame #frame_id
            num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            for frame_id in range(num_frames):
                _, frame = vidcap.read()

                if frame_id >= 1755 and frame_id <= 1835:
                    frame = draw_boxes(frame, gt[frame_id], color='g')
                    frame = draw_boxes(frame, det[frame_id], color='b', det=True)

                    cv2.imshow('frame', frame)
                    if cv2.waitKey() == 113:  # press q to quit
                        break

            cv2.destroyAllWindows()
