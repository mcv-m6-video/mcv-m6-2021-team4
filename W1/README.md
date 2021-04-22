# Week 1
 
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

## Execution
 
Execute the program as follows:
 ```bash
 python w1.py -h
 
 usage: w1.py [-h] [--t1] [--t2] [--t3] [--t4]

Video Surveillance for Road Traffic Monitoring. MCV-M6-Project, Team 4

optional arguments:
  -h, --help  show this help message and exit
  --t1        execute task 1: generate noisy boxes from annotations and
              compute AP/mIoU
  --t2        execute task 2: compute AP/mIoU vs frame (temporal) for a
              specific detector
  --t3        execute task 3: compute MSEN, PEPN, and visualize the errors
  --t4        execute task 4: visualize optial flow (two methods)
```
Note: Add AIcity folder into data/ to use it
with annotations file 'ai_challenge_s03_c010_full_annotation.xml' in path 'AICity_data/train/S03/c010'
