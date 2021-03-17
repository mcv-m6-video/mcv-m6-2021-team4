# Week 2

* Task 1: Single gaussian modelling, static (non-adaptive) background estimation.
* Task 2: Single gaussian modelling, adaptive background estimation.
* Task 3: Compare SOTA methods for background subtraction.
* Task 4: Multipe gaussian modelling in different color spaces.

## Execution
 
Execute the program (from mcv-m6-2021-team4/, NOT from mcv-m6-2021-team4/W2) as follows:
 ```bash
 python w2.py -h
 
 usage: w2.py [-h] [--t1] [--t2] [--t3] [--t4] [--video_path VIDEO_PATH]
             [--roi_path ROI_PATH] [--gt_path GT_PATH]
             [--results_path RESULTS_PATH] [--num_frames_eval NUM_FRAMES_EVAL]
             [--bg_est BG_EST] [--alpha ALPHA] [--rho RHO] [--show_boxes]
             [--color_space COLOR_SPACE] [--voting VOTING] [--save_results]
             [--sota_method {MOG,MOG2,LSBP,KNN,GSOC}]

Video Surveillance for Road Traffic Monitoring. MCV-M6-Project, Team 4

optional arguments:
  -h, --help            show this help message and exit
  --t1                  execute task 1: static method to estimate backround
  --t2                  execute task 2: adaptive method to estimate backround
  --t3                  execute task 3: SOTA methods
  --t4                  execute task 4: color method to estimate background
  --video_path VIDEO_PATH
                        path to video
  --roi_path ROI_PATH   path to roi
  --gt_path GT_PATH     path to annotations
  --results_path RESULTS_PATH
                        path to save results
  --num_frames_eval NUM_FRAMES_EVAL
                        number of frames to evaluate
  --bg_est BG_EST       bg estimation method
  --alpha ALPHA         alpha parameter
  --rho RHO             rho parameter
  --show_boxes          show bounding boxes
  --color_space COLOR_SPACE
                        color space
  --voting VOTING       voting method
  --save_results        save detections
  --sota_method {MOG,MOG2,LSBP,KNN,GSOC}
                        State of the art method for Background Substraction
```
Note: Add AIcity folder into data/ to use it
