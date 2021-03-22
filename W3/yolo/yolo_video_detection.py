from models import *
from utils import *
import cv2

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from tqdm import tqdm

def detect_image(model,img,params):
    # scale and pad image
    ratio = min(params['img_size']/img.size[0], params['img_size']/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(torch.cuda.FloatTensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, params['conf_thres'], params['nms_thres'])
    return detections[0]

def yolo_to_ai(frame_id,detections,results_path):
    # yolo format: [left,top,right,bottom,obj_confidence, class_confidence, class]
    # desired format: [frame_id,-1,bb_left,bb_top,bb_width,bb_height,conf]
    frame_detections=[]
    coco_car_id = 2
    ratio = 1920/416
    for detection in detections:
        if int(detection[6]) == coco_car_id :
            tlx = ratio*float(detection[0])
            tly = ratio*(float(detection[1])-91) # 416/1920*(imw-imh)/2 = 91
            w = ratio*(float(detection[2])-float(detection[0]))
            h = ratio*(float(detection[3])-float(detection[1]))
            det = str(frame_id+1)+',-1,'+str(tlx)+','+str(tly)+','+str(w)+','+str(h)+','+str((float(detection[5])+float(detection[4]))/2)+',-1,-1,-1\n'
            frame_detections.append(det)
            with open(results_path+'detections.txt', 'a+') as f:
                f.write(det)
    return frame_detections

def yolo_inference(params):
    # Load model and weights
    model = Darknet(params['config_path'], img_size=params['img_size'])
    model.load_weights(params['weights_path'])
    model.cuda()
    model.eval()
    classes = load_classes(params['class_path'])

    vid = cv2.VideoCapture(params['video_path'])
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_detections=[]
    for frame_id in tqdm(range(num_frames)):
        _, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(model,pilimg,params)
        frame_detections = yolo_to_ai(frame_id,detections,params['results_path'])
        vid_detections.append(frame_detections)

    return vid_detections

if __name__ == '__main__':
	params = {
	    'video_path': '/mnt/gpid08/users/ian.riera/AICity_data/train/S03/c010/vdo.avi',
	    'config_path': 'yolov3.cfg',
	    'weights_path': 'yolov3.weights',
	    'results_path': './',
	    'class_path': 'coco.names',
	    'img_size': 416,
	    'conf_thres': 0.5,
	    'nms_thres': 0.4,
	}
	# print(yolo_inference(params))
	yolo_inference(params)
