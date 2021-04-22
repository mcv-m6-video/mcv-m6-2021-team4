# Week 3

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


## Execution
 
To execute each task, simply run with:

```bash
python task#.py -h
```

TODO: Each task has a different number of input arguments and it's executed from a different file. We must refactor it to have a single main function.

Note: Add AIcity folder into data/ to use it. Clone [this repo](https://github.com/abewley/sort) in ./W3 to use the SORT algorithm. Clone [this repo](https://github.com/matterport/Mask_RCNN) in ./W3/mask_rcnn to use Mask R-CNN (Keras), and download the official YOLOv3 weights, coco.classes and coco.names from [here](https://pjreddie.com/darknet/yolo/) and include them in ./W3/yolo.
