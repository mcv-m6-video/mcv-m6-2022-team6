# Week 4
## Summary 

## Tasks
### Task 1 Optical Flow
This task is divided in 3 steps

**Task 1.1** Optical flow with Block Matching, here we implement an iterative method that match regions from the previous frame to the new one. That comparative is done using different methods and parameters.

* In order to visualize all this experiments run the command task4_1_1() passing as a first argument methods/area_block/steps to visualize this options for example

> python task4_1_1("methods")

**Task 1.2** Off-the-shelt Optical Flow, on this task we implement the methods Horn Schunk, Farneback 2003 and Lucas-Kanade.

* In order to visualize this different methods run the command task4_1_2() passing as the first argument Shunk (for Horn Schunk), Farneback or Kanade (for Lucas-Kanade)

### Task 2 Multi-target single camera tracking
* Applying the best method  on all the cameras from Sequence S01, S03, and S04 of the AI City Challenge Track 1
* Report best results for each camera.

To track the cars of a video sequence and evaluate the tracking (IDF1), run:

python W4/task_2.py -h

usage: task_2.py [-h] [--track_method {overlap,kalman}]
                      [--det_method {retina,mask,ssd,yolo}]
                      [--det_dir DET_DIR] [--data_path DATA_PATH]
                      [--seqs SEQS] [--show_boxes] [--save_filtered]

optional arguments:
  -h, --help            show this help message and exit
  --track_method {overlap,kalman}
                        method used to track cars
  --det_method {retina,mask,ssd,yolo}
                        load detections obtained with this method
  --det_dir DET_DIR     path from where to load detections
  --data_path DATA_PATH
                        path to sequences of AICity
  --seqs SEQS           sequence/camera from AICity dataset
  --show_boxes          show bounding boxes
  --save_filtered       save filtered detections (without parked cars)


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
