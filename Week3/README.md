# Week 3
## Summary 

The goal of this week is learn about using different Object Detection methods and fine tune it.

### Requirements

To train this code, is necessary run it with a computer with GPU or do it in a google colab.

## Tasks
### Task 1 Object detection
This task is divided in 2 steps
Both task can be visualized with Task1_3.ipynb jupyter notebook.

**Task 1.1** Off-the-shelf, for this task we were reqeusted to implment one or more object detection like Detectron2, Pytorch or Keras.

* The implementation of this experiments are done in a jupyter notebook called TASK1_3.ipynb


**Task 1.2** Fine-tune, as previous weeks once method is implemented we must tune it to our data.


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
