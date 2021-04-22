# Week 4

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

## Execution
 
To execute each task, simply run with:

```bash
python task#.py -h
```

TODO: Each task has a different number of input arguments and it's executed from a different file. We must refactor it to have a single main function.

Note: Add AIcity folder into data/ to use it. Clone the following repos to use each algorithm:

+ [Pyflow](https://github.com/pathak22/pyflow)
+ [RAFT](https://github.com/oscar-lorente/RAFT)
+ [MaskFlownet](https://github.com/oscar-lorente/MaskFlownet)
+ [Mesh-Flow](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization)
+ [L1 Video stabilization](https://github.com/VAIBHAV-2303/VideoStabilization)
