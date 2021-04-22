import os, sys, cv2, torch, torchvision, random, argparse
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

import matplotlib as plt

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.distances import CosineSimilarity

from reid_utils import get_id_frames_cam, get_data_loader, compare_cams, merge_dicts, invert_dict, load_trunk_embedder
from eval_tracking_mtmc import evaluate_mtmc

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_csv', type=str, default='./W5/det/det_car_patches_annotations.csv',
                        help='path to gt csv containing annotations')
    parser.add_argument('--det_patches', type=str, default='./W5/det/det_patches/',
                        help='path to gt folder containing car patches')
    parser.add_argument('--trunk_model', type=str, default='./W5/task2/model/0001_32_256/trunk.pth',
                        help='path to trunk model')
    parser.add_argument('--embedder_model', type=str, default='./W5/task2/model/0001_32_256/embedder.pth',
                        help='path to embedder model')
    parser.add_argument('--save_reid', type=str, default='./W5/det/reid',
                        help='path to save reid detections')
    parser.add_argument('--show_reid', action='store_true',
                        help='show example of reid')
    parser.add_argument('--eval_mtmc', action='store_true',
                        help='evaluate multi target multi camera tracking')
    parser.add_argument('--thr', type=float, default=0.6,
                        help='threshold to consider a match')
    parser.add_argument('--patches_to_compare_c1', type=int, default=3,
                        help='number of patches to compare with cam1')
    parser.add_argument('--patches_to_compare_c2', type=int, default=5,
                        help='number of patches to compare with cam2')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    trunk, embedder = load_trunk_embedder(args.trunk_model, args.embedder_model)

    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=args.thr)
    inference_model = InferenceModel(trunk, embedder, match_finder=match_finder, batch_size=64)

    labels = pd.read_csv(args.det_csv)
    test_data, _ = get_data_loader('test', labels, args.det_patches)
    indices_cameras = c_f.get_labels_to_indices(test_data.camera)

    id_frames_cams = {}
    for cam in ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']:
        id_frames_cams[cam] = get_id_frames_cam(test_data, indices_cameras, cam)

    id_frames_all = deepcopy(id_frames_cams['c010'])

    for cam in tqdm(['c011', 'c012', 'c013', 'c014', 'c015']):
        re_id_cam = compare_cams(test_data, deepcopy(id_frames_all),
                                 deepcopy(id_frames_cams[cam]), inference_model,
                                 args.patches_to_compare_c1,
                                 args.patches_to_compare_c2)

        id_frames_all = merge_dicts(id_frames_all, re_id_cam)

    frames_id_all = invert_dict(id_frames_all)

    det_dir = os.path.join(args.save_reid,
                            ''.join(str(args.thr).split('.')) + '_' + str(args.patches_to_compare_c1) + '_' + str(args.patches_to_compare_c2))

    if args.save_reid is not None:
        for cam, id_frames_cam in id_frames_cams.items():
            os.makedirs(os.path.join(det_dir, cam), exist_ok=True)
            det_path = os.path.join(det_dir, cam, 'overlap_reid_detections.txt')
            if os.path.isfile(det_path):
                os.remove(det_path)

            frames_cam = []
            for tmp_frames in list(id_frames_cam.values()):
                frames_cam.extend(tmp_frames)
            frames_cam = sorted(frames_cam)

            for idx in frames_cam:
                box = test_data[idx]

                # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                det = str(box[4] + 1) + ',' + str(frames_id_all[idx]) + ',' + str(box[3][0]) + ',' \
                      + str(box[3][1]) + ',' + str(box[3][2]) + ',' + str(box[3][3]) + ',' + '1' + ',-1,-1,-1\n'

                with open(det_path, 'a+') as f:
                    f.write(det)

    if args.eval_mtmc:
        summary = evaluate_mtmc(det_dir,
                      "/mnt/gpid08/users/ian.riera/AICity_data/train/S03")
        print('thr: ', args.thr, ', patches c1: ', args.patches_to_compare_c1, ', patches c2: ', args.patches_to_compare_c2)
        print(summary)

        summary.to_csv(os.path.join(det_dir, 'evaluation_mtmc.csv'), sep='\t')
