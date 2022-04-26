# Week 3
## Summary 

The goal of this week is learn about using different Object Detection methods and fine tune it.

### Requirements

To train this code, is necessary run it with a computer with GPU or do it in a google colab.

## Tasks
### Task 1 Object detection
This task is divided in 3 steps



**Task 1.1/1.2** Off-the-shelf/ Fine-tune, for this task we were reqeusted to implment one or more object detection like Detectron2, Pytorch or Keras. Also as previous weeks once the method is implemented we must fine tune it to our data.

* Task 1.1 and 1.2 can be visualized with Task1_2.ipynb jupyter notebook.

**Task 1.3** K-Fold Cross-validation, this task consist on repeat n times the training-test proces changing the testset and then picking the average to see a more realistic value of accuracy.

* To visualize this task run the jupyter notebook Task1_3.ipynb


### Task 2 Object tracking
**Task 2.1** Tracking by Operlap

**Task 2/2.1** Tracking by Overlap / Tracking with a Kalman Filter
* Apply methods of object tracking on previous detection methods
* Report best results for each camera.

**Task 2.3** IDF1 score

Implement IDF1 socre (ratio of correctly identified detection)

To track the cars of a video sequence and evaluate the tracking (IDF1), run:

```bash
python Week3/task_2.py -h

usage: Task2.py.py [-h] [--iou_th]
                        [--tracker {iou,iou_direction,kalman,kalmansort}]
                        [--input {gt,retinanet50,faster50,retinanet101,faster101}] [--preview {True, False}]

optional arguments:
  -h, --help            show this help message and exit
  --iou_th              set threshold for iou methods
  --tracker {iou,iou_direction,kalman,kalmansort}
                        method used to track cars
  --input {gt,retinanet50,faster50,retinanet101,faster101}
                        load detections obtained with this method
  --preview             chose if show or not images

```

## Results

