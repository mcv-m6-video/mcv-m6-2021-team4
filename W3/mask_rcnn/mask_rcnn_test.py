'''
READ ME!
Libraries that work:
pip install tensorflow-gpu==1.14.0
pip install keras==2.2.4

Import repo:
git clone https://github.com/matterport/Mask_RCNN


'''

import os
import sys
import random
import math
import re
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
from tqdm import tqdm
from PIL import Image
from timeit import default_timer as timer
# Root directory of the imported project
ROOT_DIR = os.path.abspath("./Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to Shapes trained weights
SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco
config = coco.CocoConfig()



def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def mask_to_ai(frame_id,detections,results_path):
    # mask format: ['rois] = [top,left, bottom,right]
    #              ['class_ids'] = class # 3 is car
    #              ['scores'] = confidence
    # desired format: [frame_id,-1,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1]
    frame_id=frame_id+1
    frame_detections=[]
    coco_car_id = 3

    for i in range(len(detections['class_ids'])):
        if detections['class_ids'][i] == coco_car_id :
            det = str(frame_id)+',-1,'+str(detections['rois'][i][1])+','+str(detections['rois'][i][0])+','+str(detections['rois'][i][3]-detections['rois'][i][1])+','+str(detections['rois'][i][2]-detections['rois'][i][0])+','+str(detections['scores'][i])+',-1,-1,-1\n'
            frame_detections.append(det)
            
            with open(results_path+'detections_mask.txt', 'a+') as f:
                f.write(det)
                
    return frame_detections

def detect_image(img,times): #scale_percent=25,
    start = timer()
    detections = model.detect([img], verbose=0)
    end = timer()

    times.append(end-start)
    return detections[0],times


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"



#------COCO class names-------
coco_car_id = 3

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#--------------------------------------


#--------Load Model--------------------
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set weights file path
if config.NAME == "shapes":
    weights_path = SHAPES_MODEL_PATH
elif config.NAME == "coco":
    weights_path = COCO_MODEL_PATH
# Or, uncomment to load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
#-------------------------------
if tf.test.gpu_device_name(): 
    print('Default GPU Device:  {}'.format(tf.test.gpu_device_name()))
#-------Run Detection------
results_path = 'PATH.TO.RESULTS'
vid = cv2.VideoCapture('PATH.TO/AICity_data/train/S03/c010/vdo.avi')

frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

times = []
#while(True):
for ii in tqdm(range(frame_count)):#frame_count
    ret, frame1 = vid.read()
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    

    # Run detection
    detections,times = detect_image(frame,times)#,scale_percent=22
    frame_detections = mask_to_ai(ii,detections,results_path)
print('Inference time (s/img): ', np.mean(times))  
