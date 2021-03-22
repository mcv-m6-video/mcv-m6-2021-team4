import sys, os, cv2
import numpy as np
import torch
assert torch.__version__.startswith("1.7")   # need to manually install torch 1.8 if Colab changes its default version
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures.instances import Instances

from tqdm import tqdm

def detect(video_path):
    save_visual_detections = False

    results_dir = 'results/task1_1_retina/'

    coco_car_id = 2
    conf_thr = 0.5

    model = 'retinanet_R_50_FPN_3x'
    model_path = 'COCO-Detection/' + model + '.yaml'
    print(model_path)

    # Run a pre-trained detectron2 model
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.OUTPUT_DIR = results_dir + model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    det_path = os.path.join(cfg.OUTPUT_DIR, 'detections.txt')
    if os.path.exists(det_path):
        os.remove(det_path)

    vidcap = cv2.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_frames = 3

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for frame_id in tqdm(range(num_frames)):
        _, frame = vidcap.read()

        start.record()
        outputs = predictor(frame)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        pred_boxes = outputs["instances"].pred_boxes.to("cpu")
        scores = outputs["instances"].scores.to("cpu")
        pred_classes = outputs["instances"].pred_classes.to("cpu")

        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []

        for idx, pred in enumerate(pred_classes):
            if pred.item() == coco_car_id and scores[idx].item() >= conf_thr:
                box = pred_boxes[idx].tensor.numpy()[0]

                # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                det = str(frame_id+1)+',-1,'+str(box[0])+','+str(box[1])+','+str(box[2]-box[0])+','+str(box[3]-box[1])+','+str(scores[idx].item())+',-1,-1,-1\n'

                with open(det_path, 'a') as f:
                    f.write(det)

                filtered_boxes.append(pred_boxes[idx].tensor.numpy()[0])
                filtered_scores.append(scores[idx].item())
                filtered_classes.append(pred_classes[idx].item())

        filtered_outputs = Instances([256,256], pred_boxes=torch.tensor(filtered_boxes), scores=torch.tensor(filtered_scores), pred_classes=torch.tensor(filtered_classes))

        if save_visual_detections:
            output_path = os.path.join(cfg.OUTPUT_DIR, 'det_frame_' + str(frame_id) + '.png')
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_instance_predictions(filtered_outputs)
            cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    print(np.mean(times))

    return det_path
