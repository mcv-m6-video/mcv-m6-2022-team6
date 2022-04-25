# Week 1

## Summary

## Tasks

<details>
<summary>Task 1.1 Dataset metrics</summary>
 <p>This task is presented to make us familiar with dataset. The aim of this 
  
   To run this task you must call task101() or just call main_all_tasks.py() </p>
</details>

<details>
<summary>Task 1.2 Detection metrics. Temporal analysis</summary>
 <p>Content 1 
  
   To call this task you must run task102 Content 1
   Content 1 
   Content 1</p>
</details>

<details>
<summary>Task 2.1 Optical flow evaluation metrics</summary>
 <p>Content 1 
  
   Content 1
   Content 1 
   Content 1</p>
</details>

<details>
<summary>Task 2.2 Visual representation optical flow</summary>
 <p>Content 1 
  
   Content 1
   Content 1 
   Content 1</p>
</details>


# Referenia 
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

**Task 2/2.1**
* Applying the best method  on all the cameras from Sequence S01, S03, and S04 of the AI City Challenge Track 1
* Report best results for each camera.

To track the cars of a video sequence and evaluate the tracking (IDF1), run:

```bash
python Week4/task_2.py -h

usage: task_2.py.py [-h] [--track_method {overlap,kalman}]
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

```
