import cv2
import sys

sys.path.append("W2")
from bg_estimation import train, eval

def run(args, bg_est='static', alpha=3, rho=0.05, color_space='RGB', voting='simple'):
    params = {
        'video_path': args.video_path,
        'roi_path': args.roi_path,
        'gt_path': args.gt_path,
        'results_path': args.results_path,
        'num_frames_eval': args.num_frames_eval,
        'show_boxes': args.show_boxes,
        'save_results': args.save_results,
        'bg_est': bg_est,
        'alpha': alpha,
        'rho': rho,
        'color_space': color_space,
        'voting': voting  # simple, unanimous
    }

    vidcap = cv2.VideoCapture(params['video_path'])
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = [frame_height,frame_width]
    print("Total frames: ", frame_count)

    train_len = int(0.25 * frame_count)
    test_len = frame_count - train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    # Train
    mean, std = train(vidcap, frame_size, train_len, params)

    # Evaluate
    return eval(vidcap, frame_size, mean, std, params)