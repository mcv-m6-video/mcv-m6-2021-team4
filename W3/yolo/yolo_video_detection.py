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

def detect_image(img,params):
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
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, params['conf_thres'], params['nms_thres'])
    return detections[0]

def yolo_to_ai(frame_id,detections,results_path):
    # yolo format: [left,top,right,bottom,obj_confidence, class_confidence, class]
    # desired format: [frame_id,-1,bb_left,bb_top,bb_width,bb_height,conf]
    frame_id=frame_id+1
    frame_detections=[]
    coco_car_id = 2
    for detection in detections:
        if int(detection[6]) == coco_car_id :
            det = str(frame_id)+',-1,'+str(float(detection[0]))+','+str(float(detection[1]))+','+str(float(detection[2])-float(detection[0]))+','+str(float(detection[3])-float(detection[1]))+','+str((float(detection[5])+float(detection[4]))/2)+',-1,-1,-1\n'
            frame_detections.append(det)
            with open(results_path+'yolo_detections.txt', 'a+') as f:
                f.write(det)
    return frame_detections

def yolo_inference(params):
	# Load model and weights
	model = Darknet(params['config_path'], img_size=params['img_size'])
	model.load_weights(params['weights_path'])
	model.cuda()
	model.eval()
	classes = load_classes(params['class_path'])
	Tensor = torch.cuda.FloatTensor

	vid = cv2.VideoCapture(params['video_path'])
	frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

	vid_detections=[]
	for frame_id in range(frame_count):
	    ret, frame = vid.read()
	    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	    pilimg = Image.fromarray(frame)
	    detections = detect_image(pilimg,params)
	    frame_detections = yolo_to_ai(frame_id,detections,params['results_path'])
	    vid_detections.append(frame_detections)

	return vid_detections


def run():

	params = {
	    'video_path': 'video_path',
	    'config_path': 'Path.To.yolov3.cfg',
	    'weights_path': 'Path.To.yolov3.weights',
	    'results_path': 'results_path',
	    'class_path': 'Path.To.coco.names',
	    'img_size': 416,
	    'conf_thres': 0.5,
	    'nms_thres': 0.4,
	}
	print(yolo_inference(params))
