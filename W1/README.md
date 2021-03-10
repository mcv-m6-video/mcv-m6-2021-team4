# Week 1

* Task 1: Detection metrics.
* Task 2: Detection metrics. Temporal analysis.
* Task 3: Optical flow evaluation metrics.
* Task 4: Visual representation optical flow.

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
