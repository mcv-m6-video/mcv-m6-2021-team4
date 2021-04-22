# Week 2

The goal of this project is to:
* **Background estimation**
  * Model the background pixels of a video sequence using a simple statistical model to classify the background / foreground    
    + Adaptive / Non-adaptive 
    + Single Gaussian per pixel
    
* **Comparison with more complex models**

#### Task 1: Gaussian distribution and Evaluate
+ One Gaussian function to model each background pixel
  + First 25% of the video sequence to model background
  + Mean and variance of pixels
 + Second 75% to segment the foreground and evaluate
+ Evaluate results

#### Task 2: Recursive Gaussian modeling. Evaluate and compare to non-recursive
+ Adaptive modelling
  + First 25% frames for training
  + Second 75% left background adapts
+ Best pair of values (𝛼, ⍴) to maximize mAP
  + Non recursive search
  + Grid search

#### Task 3: Compare with state-of-the-art and Evaluation
+ P. KaewTraKulPong et.al. An improved adaptive background mixture model for real-time tracking with shadow detection. In Video-Based Surveillance Systems, 2002. Implementation: BackgroundSubtractorMOG (OpenCV)
+ Z. Zivkovic et.al. Efficient adaptive density estimation per image pixel for the task of background subtraction, Pattern Recognition Letters, 2005. Implementation: BackgroundSubtractorMOG2 (OpenCV)
+ L. Guo, et.al. Background subtraction using local svd binary pattern. CVPRW, 2016. Implementation: BackgroundSubtractorLSBP (OpenCV)
+ Zivkovic, Zoran & Van der Heijden, F.. (2006). Efficient adaptive density estimation per image pixel for the task of background subtraction. Pattern Recognition Letters. 27. 773-780. 10.1016/j.patrec.2005.11.005. 

#### Task 4: Color sequences
+ Use multiple gaussians in different color spaces


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
