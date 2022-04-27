# Week 5
## Summary 

## Tasks
### Task 1 Multi-target single-camera tracking

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
### Task 2 Multi-target multi-camera tracking

#### Train Siamese Network

To train the siamese network with the car patches generated with patches_crop.py, run:

```bash
python Week5/siamese_train.py -h

usage: siamese_train.py [-h] [--gt_csv GT_CSV] [--gt_patches GT_PATCHES]
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
python Week5/re_id.py -h

usage: reid.py [-h] [--det_csv DET_CSV] [--det_patches DET_PATCHES]
               [--trunk_model TRUNK_MODEL] [--embedder_model EMBEDDER_MODEL]
               [--save_reid SAVE_REID] [--show_reid] [--eval_mtmc] [--thr THR]
               [--patches_comp_c1 PATCHES_COMP_C1]
               [--patches_comp_c2 PATCHES_COMP_C2]

optional arguments:
  -h, --help            show this help message and exit
  --det_csv DET_CSV     path to gt csv containing annotations
  --det_patches DET_PATCHES
                        path to det folder containing car patches
  --trunk_model TRUNK_MODEL
                        path to trunk model
  --embedder_model EMBEDDER_MODEL
                        path to embedder model
  --thr THR             threshold to consider a match
  --patches_to_compare_c1 PATCHES_TO_COMPARE_C1
                        number of patches to compare with cam1
  --patches_to_compare_c2 PATCHES_TO_COMPARE_C2
                        number of patches to compare with cam2
  --show_reid           show example of reid
  --save_reid SAVE_REID
                        path to save reid detections
  --eval_mtmc           evaluate multi target multi camera tracking
```
