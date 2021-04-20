import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import csv

sys.path.append("W1")
sys.path.append("W2")
sys.path.append("W3")
sys.path.append("W4")
sys.path.append("W3/sort")
from aicity_reader import read_annotations, read_detections, group_by_frame

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def extract_car_patches(vidcap, test_len, gt_path, train_path, car_patches_path, sequence, camera):

    print("Extracting Car Patches")  
    gt = read_detections(gt_path, grouped=True)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    first_frame_id = frame_id
    print(frame_id)

    for t in tqdm(range((test_len) - first_frame_id)):

        _ ,frame = vidcap.read()
        # cv2.imshow('Frame', frame)
        # keyboard = cv2.waitKey(30)

        if frame_id in gt:
            gt_bboxes = gt[frame_id]

            for box in gt_bboxes:
                # print(box)
                crop_img = frame[int(box.ytl):int(box.ybr), int(box.xtl):int(box.xbr)]
                cv2.imwrite(car_patches_path + f"/{str(box.id)}_{sequence}_{camera}_{str(frame_id)}.jpg", crop_img.astype(int))


        frame_id += 1
    
def create_csv_car_patches(vidcap, test_len, gt_path, train_path, car_patches_path, sequence, camera, writer):
    print("Create CSV car_patches_annotations.csv")  
    gt = read_detections(gt_path, grouped=True)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    first_frame_id = frame_id

    for t in tqdm(range((test_len) - first_frame_id)):

        if frame_id in gt:
            gt_bboxes = gt[frame_id]

            for box in gt_bboxes:
                # print(f"{str(box.id)}_{sequence}_{camera}_{str(frame_id)}.jpg")
                # print(str(box.id))
                writer.writerow([f"{str(box.id)}_{sequence}_{camera}_{str(frame_id)}.jpg", str(box.id)])

        frame_id += 1


if __name__ == "__main__":

    train_path = './aic19-track1-mtmc-train/train'  
    car_patches_path = './aic19-track1-mtmc-train/car_patches'
    
    if not os.path.exists(car_patches_path):
        print("Create dir")
        os.mkdir(car_patches_path)

    with open('car_patches_annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["FILENAME", "ID"]) #header


        for sequence in mylistdir(train_path):
            # print(sequence)
            for camera in mylistdir(os.path.join(train_path, sequence)):
                print(camera)
                gt_path = os.path.join(train_path, sequence, camera, 'gt', 'gt.txt')
                video_path = os.path.join(train_path, sequence, camera, 'vdo.avi')

                vidcap = cv2.VideoCapture(video_path)
                frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Total frames: ", frame_count)


                print("--------------- VIDEO ---------------")
                print(sequence, camera)
                print("--------------------------------------")


                # extract_car_patches(vidcap, frame_count, gt_path, train_path, car_patches_path, sequence, camera)
                create_csv_car_patches(vidcap, frame_count, gt_path, train_path, car_patches_path, sequence, camera, writer)


   