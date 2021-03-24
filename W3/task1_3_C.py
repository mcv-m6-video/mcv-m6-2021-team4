# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
from sklearn.model_selection import KFold
import os, json, cv2, random, sys
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import argparse

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

sys.path.append('../W1')
from aicity_reader import read_annotations
import bounding_box

from LossEvalHook import *
from MyTrainer import *

def parse_annotation(annotations):
    objs = []
    for annot in annotations:

        bbox = [annot.xtl, annot.ytl, annot.xbr, annot.ybr]

        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0
        }

        objs.append(obj)

    return objs

def create_splits(X, k=3):
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    train_folds= []
    val_folds = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        train_folds.append(X_train)
        val_folds.append(X_val)

    return train_folds, val_folds


# def load_dataset(type, set_config, thing_classes, map_classes):
def load_dataset(type, thing_classes, set_config):

    frames_path = '/mnt/gpid08/users/ian.riera/AICity_data/frames'
    gt_path = '/mnt/gpid08/users/ian.riera/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'

    gt = read_annotations(gt_path, grouped=True, use_parked=True)

    num_frames = 2141

    frame_list = np.arange(0,2141)

    np.random.seed(42)

    # Strategy C
    train_val_frames = np.random.choice(frame_list,int(len(frame_list)*0.25),replace=False)

    train_folds, val_folds = create_splits(train_val_frames, 3)

    train_fold = train_folds[set_config]
    val_fold = val_folds[set_config]

    dataset_dicts = []
    for frame_id in train_val_frames:
        if frame_id in train_fold and type == 'train' or frame_id in val_fold and type != 'train':
            record = {}
            filename = os.path.join(frames_path, str(frame_id) + '.png')

            record["file_name"] = filename
            # record["image_id"] = i+j
            record["image_id"] = filename
            record["height"] = 1080
            record["width"] = 1920

            record["annotations"] = parse_annotation(gt[frame_id])
            dataset_dicts.append(record)

    return dataset_dicts

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='faster_rcnn',
                        choices=['faster_rcnn', 'retinanet'],
                        help='pre-trained detectron2 model')

    parser.add_argument('--set_config', type=int, default=0,
                        help='cross val set combination')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--iter', type=int, default=5000,
                        help='max iterations (epochs)')

    parser.add_argument('--batch', type=int, default=512,
                        help='batch size')

    # parser.add_argument('--set_config', type=str, default='0',
    #                     help='which configuration of cross validation to use')

    return parser.parse_args(args)

if __name__ == "__main__":

    args = parse_args()

    model = 'COCO-Detection/' + args.model + '_R_50_FPN_3x.yaml'
    print('[INFO] Using model: ', model)

    train = True
    evaluate = False

    if train:
        ###-------TRAIN-----------------------------
        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

        cfg.OUTPUT_DIR = 'results/task1_3/strategyC/' + args.model + '/lr_' + str(args.lr).replace('.', '_') + '_iter_' + str(args.iter) + '_batch_' + str(args.batch) + '/' + str(args.set_config) + '/'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        thing_classes = ['Car']
        dataset='AICity'

        for d in ['train', 'val', 'test']:
            DatasetCatalog.register(dataset + '_' + d, lambda d=d: load_dataset(d, thing_classes, args.set_config))
            MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)

        metadata = MetadataCatalog.get(dataset + '_train')

        cfg.DATASETS.TRAIN = (dataset + '_train',)
        cfg.DATASETS.VAL = (dataset + '_val',)
        cfg.DATASETS.TEST = (dataset + '_test',)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.SOLVER.IMS_PER_BATCH = 2

        cfg.SOLVER.BASE_LR = args.lr
        cfg.SOLVER.MAX_ITER = args.iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # cfg.TEST.EVAL_PERIOD = 0 #eval_period
        # trainer = MyTrainer(cfg)
        # trainer.resume_or_load(resume=False)
        # trainer.train()

        ###-------INFERENCE AND EVALUATION---------------------------
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

        ### MAP #####
        #We can also evaluate its performance using AP metric implemented in COCO API.
        evaluator = COCOEvaluator(dataset + '_val', cfg, False, output_dir=cfg.OUTPUT_DIR)
        test_loader = build_detection_test_loader(cfg, dataset + '_val')
        print('------------------------ Evaluating model ' + model + ' on validation set ---------------------------------')
        print(inference_on_dataset(trainer.model, test_loader, evaluator))
        print('---------------------------------------------------------')


    if evaluate:
        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

        results_dir = 'results/task1_3/strategyC/test_inference/'
        cfg.OUTPUT_DIR = results_dir + args.model + '/'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        thing_classes = metadata.thing_classes
        dataset='AICity'

        for d in ['train', 'test']:
            DatasetCatalog.register(dataset + '_' + d, lambda d=d: load_dataset(d,thing_classes))
            MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)

        metadata = MetadataCatalog.get(dataset + '_train')

        cfg.DATASETS.TRAIN = (dataset + '_train',)
        cfg.DATASETS.TEST = (dataset + '_test',)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)

        evaluator = COCOEvaluator(dataset + '_test', cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, dataset + '_test')
        print('Evaluation with model ', model)
        print(inference_on_dataset(trainer.model, val_loader, evaluator))
