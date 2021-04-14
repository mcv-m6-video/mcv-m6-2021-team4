# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import argparse

sys.path.append('../W1')
from aicity_reader import read_annotations

from MyTrainerAugm import *

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

# def load_dataset(type, set_config, thing_classes, map_classes):
def load_dataset(type, thing_classes):

    frames_path = '/mnt/gpid08/users/ian.riera/AICity_data/frames'
    gt_path = '/mnt/gpid08/users/ian.riera/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'

    gt = read_annotations(gt_path, grouped=True, use_parked=True)

    num_frames = 2141

    train_val_split = 0.67
    # train_val_split = 1.0

    if type == 'train':
        init_frame = 0
        last_frame = int(train_val_split*(int(num_frames*0.25)))
    elif type == 'val' or type == 'test':
        init_frame = int(train_val_split*(int(num_frames*0.25)))
        last_frame = int(num_frames*0.25)
    # elif type == 'test':
    #     init_frame = int(num_frames*0.25)
    #     last_frame = num_frames

    dataset_dicts = []
    for frame_id in range(init_frame, last_frame):
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

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--iter', type=int, default=5000,
                        help='max iterations (epochs)')

    parser.add_argument('--batch', type=int, default=512,
                        help='batch size')

    parser.add_argument('--augm', action='store_true',
                        help='use augmentation')

    parser.add_argument('--freeze', type=int, default=2,
                        help='stages of ResNet to freeze')

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

        cfg.OUTPUT_DIR = 'results/task1_2/' + args.model + '/augm/lr_' + str(args.lr).replace('.', '_') + '_iter_' + str(args.iter) + '_batch_' + str(args.batch) + '/'  # + args.set_config + '/'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        thing_classes = ['Car']
        dataset='AICity'

        for d in ['train', 'val', 'test']:
            DatasetCatalog.register(dataset + '_' + d, lambda d=d: load_dataset(d, thing_classes))
            MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)

        metadata = MetadataCatalog.get(dataset + '_train')

        cfg.DATASETS.TRAIN = (dataset + '_train',)
        cfg.DATASETS.VAL = (dataset + '_val',)
        cfg.DATASETS.TEST = (dataset + '_test',)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.SOLVER.IMS_PER_BATCH = 2

        cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze

        cfg.SOLVER.BASE_LR = args.lr
        cfg.SOLVER.MAX_ITER = args.iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch

        #eval_period: frequence of validation loss computations (to plot curves)
        cfg.TEST.EVAL_PERIOD = 0

        if args.augm:
            trainer = MyTrainerAugm(cfg)
        else:
            trainer = MyTrainer(cfg)
            
        trainer.resume_or_load(resume=False)
        trainer.train()

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

        results_dir = 'results/task1_2/test_inference/'
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
