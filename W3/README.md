# Week 2

* Task 1_1: Object Detection Off-the-shelf
* Task 1_2: Fine tune architectures on AICity dataset
* Task 1_3: Fine tune using cross validation
* Task 2_1: Object tracking using IOU (max overlap)
* Task 2_2: Object tracking using Kalman filter
* Task 3: Object detection and tracking on the official AICity challenge dataset (not included in this repo, as it's only use the implemented alrogithms with another video)

## Execution
 
To execute each task, simply run with:

```bash
python task#.py -h
```

TODO: Each task has a different number of input arguments and it's executed from a different file. We must refactor it to have a single main function.

Note: Add AIcity folder into data/ to use it. Clone [this repo](https://github.com/abewley/sort) in ./W3 to use the SORT algorithm. Clone [this repo](https://github.com/matterport/Mask_RCNN) in ./W3/mask_rcnn to use Mask R-CNN (Keras), and download the official YOLOv3 weights, coco.classes and coco.names from [here](https://pjreddie.com/darknet/yolo/) and include them in ./W3/yolo.
