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


def create_csv_patches(video_path, det_path, patches_path, sequence, camera, writer):
    vidcap = cv2.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", num_frames)

    det = read_detections(det_path, grouped=True)

    for frame_id in tqdm(range(num_frames), desc='Creating patches of seq ' + sequence + '/' + camera):
        _, frame = vidcap.read()
        if frame_id in det:
            det_bboxes = det[frame_id]

            for box in det_bboxes:
                # crop_img = frame[int(box.ytl):int(box.ybr), int(box.xtl):int(box.xbr)]
                # cv2.imwrite(patches_path + f"/{str(box.id)}_{sequence}_{camera}_{str(frame_id)}.jpg",
                #             crop_img.astype(int))

                filename = str(box.id) + '_' + sequence + '_' + camera + '_' + str(frame_id) + '.jpg'
                writer.writerow([filename, str(box.id), sequence, camera, str(frame_id), str(box.xtl), str(box.ytl),
                                 str(box.xbr), str(box.ybr), str(box.center[0]), str(box.center[1])])

        frame_id += 1

    return


if __name__ == "__main__":
    sequence = 'S03'
    videos_path = os.path.join('./data/AICity_data/train',
                               sequence)
    detections_path = os.path.join('./W5/task1_1/detections/faster',
                                   sequence)
    patches_path = './patches'
    os.makedirs(patches_path, exist_ok=True)

    with open('car_patches_annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["FILENAME", "ID", "SEQUENCE", "CAMERA", "FRAME", "XTL", "YTL", "XBR", "YBR", "CENTER_X", "CENTER_Y"])  # header

        for camera in sorted(mylistdir(detections_path)):
            det_path = os.path.join(detections_path, camera, 'overlap_filtered_detections.txt')
            video_path = os.path.join(videos_path, camera, 'vdo.avi')

            print("--------------- VIDEO ---------------")
            print(sequence, camera)
            print("--------------------------------------")

            create_csv_patches(video_path, det_path, patches_path, sequence, camera, writer)
