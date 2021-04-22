# Week 5

#### Task 1: Multi-target single camera tracking

* Applying the best method from last week on all the cameras from Sequence S03 of the AI City Challenge Track 3
* Report best results for each camera.

#### Task 2: Multi-target multi camera tracking

* Using Metric Learning to cluster ID's of different cameras.
* Developing a ReID algorithm for a cross camera tracking.

## Execution

### MTSC

#### Car Detection
 
To detect the cars of a video sequence, run:

```bash
python task1/detect.py -h

usage: detect.py [-h] [--method {faster,mog}] [--seq SEQ]
                 [--data_path DATA_PATH] [--results_dir RESULTS_DIR]
                 [--show_boxes] [--save_results]

optional arguments:
  -h, --help            show this help message and exit
  --method {faster,mog}
  --seq SEQ             sequence/camera from AICity dataset
  --data_path DATA_PATH
                        path to sequences of AICity
  --results_dir RESULTS_DIR
                        path to save results
  --show_boxes          show bounding boxes
  --save_results        save detection

```

#### Tracking and Evaluation
 
To track the cars of a video sequence and evaluate the tracking (IDF1), run:

```bash
python W5/task1/eval_tracking.py -h

usage: eval_tracking.py [-h] [--track_method {overlap,kalman}]
                        [--det_method {faster,mog,mask,ssd,yolo}]
                        [--det_dir DET_DIR] [--data_path DATA_PATH]
                        [--seqs SEQS] [--show_boxes] [--save_filtered]

optional arguments:
  -h, --help            show this help message and exit
  --track_method {overlap,kalman}
                        method used to track cars
  --det_method {faster,mog,mask,ssd,yolo}
                        load detections obtained with this method
  --det_dir DET_DIR     path from where to load detections
  --data_path DATA_PATH
                        path to sequences of AICity
  --seqs SEQS           sequence/camera from AICity dataset
  --show_boxes          show bounding boxes
  --save_filtered       save filtered detections (without parked cars)

```

### MTMC

#### Train Siamese Network

To train the siamese network with the car patches generated with utils/crop_patches.py, run:

```bash
python W5/task2/train_siamese.py -h

usage: train_siamese.py [-h] [--gt_csv GT_CSV] [--gt_patches GT_PATCHES]
                        [--save_model SAVE_MODEL] [--epochs EPOCHS] [--lr LR]
                        [--batch_size BATCH_SIZE] [--embeddings EMBEDDINGS]

optional arguments:
  -h, --help            show this help message and exit
  --gt_csv GT_CSV       path to gt csv containing annotations
  --gt_patches GT_PATCHES
                        path to gt folder containing car ptches
  --save_model SAVE_MODEL
                        path to save trained model
  --epochs EPOCHS       number of epochs to train
  --lr LR               learning rate
  --batch_size BATCH_SIZE
                        batch size
  --embeddings EMBEDDINGS
                        number of embeddings
```

#### Re-Identification

To re-assign IDs for each car in different cameras, run:

```bash
python W5/task2/reid.py -h

usage: reid.py [-h] [--det_csv DET_CSV] [--det_patches DET_PATCHES]
               [--trunk_model TRUNK_MODEL] [--embedder_model EMBEDDER_MODEL]
               [--save_reid SAVE_REID] [--show_reid] [--eval_mtmc] [--thr THR]
               [--patches_to_compare_c1 PATCHES_TO_COMPARE_C1]
               [--patches_to_compare_c2 PATCHES_TO_COMPARE_C2]

optional arguments:
  -h, --help            show this help message and exit
  --det_csv DET_CSV     path to gt csv containing annotations
  --det_patches DET_PATCHES
                        path to gt folder containing car patches
  --trunk_model TRUNK_MODEL
                        path to trunk model
  --embedder_model EMBEDDER_MODEL
                        path to embedder model
  --save_reid SAVE_REID
                        path to save reid detections
  --show_reid           show example of reid
  --eval_mtmc           evaluate multi target multi camera tracking
  --thr THR             threshold to consider a match
  --patches_to_compare_c1 PATCHES_TO_COMPARE_C1
                        number of patches to compare with cam1
  --patches_to_compare_c2 PATCHES_TO_COMPARE_C2
                        number of patches to compare with cam2
```

Note: Add AIcity folder into data/ to use it. Clone the following repos to use each algorithm:

+ [Pytorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

