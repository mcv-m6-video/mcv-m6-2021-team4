import cv2
import sys
sys.path.append("W1")
from bg_estimation import train_sota, eval_sota

def run(args):
    params = {
        'video_path': args.video_path,
        'roi_path': args.roi_path,
        'gt_path': args.gt_path,
        'show_boxes': args.show_boxes,
        'sota_method': args.sota_method
    }

    vidcap = cv2.VideoCapture(params['video_path'])
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", frame_count)

    train_len = int(0.25*frame_count)
    test_len = frame_count-train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    print("Background Substractor Method: ", args.sota_method)

    if params["sota_method"] == 'MOG':
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=2, backgroundRatio=0.7)
    elif params["sota_method"] == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=36, detectShadows=True)
    elif params["sota_method"] == 'LSBP':
        backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif params["sota_method"] == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=True)
    elif params["sota_method"] == 'GSOC':
        backSub = cv2.bgsegm.createBackgroundSubtractorGSOC()

    backSub = train_sota(vidcap, train_len, backSub)
    return eval_sota(vidcap, test_len, backSub, params)
