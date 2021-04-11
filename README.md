# mcv-m6-2021-team4
## Team 4
## Contributors 
- [Pol Albacar](https://github.com/polalbacar) - palbacarfernandez@gmail.com
- [Òscar Lorente](https://github.com/oscar-lorente) - oscar.lorente.co@gmail.com
- [Eduard Mainou](https://github.com/EddieMG) - eduardmgobierno@gmail.com
- [Ian Riera Smolinska](https://github.com/ianriera) - ian.riera.smolinska@gmail.com

## Summary
This repository contains the code related to the project on 'Video Surveillance for Road Traffic Monitoring' of the [Module 6: Video Analysis](https://pagines.uab.cat/mcv/content/m6-video-analysis)  of the Master in Computer Vision at UAB. 
An extended README file for each week can be found in the respective folder.

## Week 1
 
The goal of this project is to:
* **Learn about the databases to be used**
    * AICityChallenge
    * KITTI  
    
* **Implement and get acquainted with the evaluation metrics**
    * Mean Intersection over Union
    * Mean Average Precision  
    * Mean Square Error in Non-occluded areas
    * Percentage of Erroneous Pixels in Non-occluded areas
    
* **Analyze:**
    * Effect of different noise additions
    * IoU vs Time
    * Optical Flow

## Week 2
 
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


## Week 3
#### Task 1: Object Detection
+ Off-the-shelf
   + Faster R-CNN - Detectron2
   + RetinaNet - Detectron2
   + YOLOv3 - Pytorch
   + Mask R-CNN - Keras
+ Fine-tune with our data
+ K-Fold Cross-Validation
   + Strategy A - First 25% for training - second 75% for test.
   + Strategy B - First 25% Train (cross-val) - last 75% Test (same as Strategy A).
   + Strategy C - Random 25% Train (cross-val) - rest for Test

#### Task 2: Object Tracking
+ Tracking by Maximum Overlap
+ Tracking with a Kalman Filter using [SORT](https://github.com/abewley/sort) and [2-D-Kalman-Filter](https://github.com/RahmadSadli/2-D-Kalman-Filter)
+ IDF1 score


## Week 4
#### Task 1: Optical Flow
+ Block Matching
   + Exhauastive search (SSD, SAD)
   + Template matching (NCC)
   + Grid search to optimize hyperparameters: motion type (forward, backward), block size (8, 16, .., 128) and search area (8, 16, .., 128)
+ Off-the-shelf
   + [Pyflow](https://github.com/pathak22/pyflow)
   + [RAFT](https://github.com/oscar-lorente/RAFT)
   + [MaskFlownet](https://github.com/oscar-lorente/MaskFlownet)
   + [Lucas Kanade](https://docs.opencv.org/3.3.1/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)

#### Task 2: Video Stabilization
+ With block matching
   + Using the mean
   + Using the median
   + Applying Savitsky-Golay filter
+ Off-the-shelf
   + [Vidstab](https://github.com/AdamSpannbauer/python_video_stab)
   + [Mesh-Flow](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization)
   + [L1 Video Stabilization](https://github.com/VAIBHAV-2303/VideoStabilization)

#### Task 3: Object Tracking
+ With Optical Flow
   + Using Lucas Kanade
   + Using Block Matching 
+ (Optional) for AI City Challenge
   + With and without optical flow
