import numpy as np
import pandas as pd


def read_detections(file_name, nframes, iou_th=0.5):
    annotations = np.empty(nframes, dtype=np.object)
    gt_ids = {}
    for i in range(nframes):
        annotations[i] = np.empty((0, 5), dtype=np.object)# creates an empy list equal to the number of frames
        gt_ids[i] = []

    detections = pd.read_csv(file_name, delimiter=",", header=None).to_numpy()

    for detection in detections:
        frameId = int(detection[0]) - 1

        if iou_th <= detection[6]:
            bbox = np.hstack((detection[2:4], detection[2:4] + detection[4:6], detection[6]))
            id = int(detection[1])
            annotations[frameId] = np.vstack([annotations[frameId], bbox])
            gt_ids[frameId].append(id)

    return gt_ids, annotations


def read_detections_withid(file_name, iou_th=0.5):
    output = {}
    detections = pd.read_csv(file_name, delimiter=",", header=None).to_numpy()
    for detection in detections:
        frameId = int(detection[0]) - 1
        if not frameId in output:
            output[frameId] = []

        if detection[6] >= iou_th:
            bbox = np.hstack((detection[2:4], detection[2:4] + detection[4:6], detection[6]))
            id = int(detection[1])
            output[frameId].append({'bbox': bbox, 'id': id})

    return output


def load_single_detections(seq, cam, det, n_frames):
    det_file = "dataset/train/%s/%s/det/det_%s.txt" % (seq, cam, det)
    return read_detections(det_file, n_frames)[1]


def load_detections(seq, cameras, det):
    detections = {}
    for i in range(cameras.shape[0]):
        cam = cameras.iloc[i]
        det_file = "dataset/train/%s/%s/det/det_%s.txt" % (seq, cam['Name'], det)
        detections[cam['Name']] = read_detections(det_file, cam['Frames'])[1]

    return detections


def load_gt_single(seq, cam, n_frames):
    gt = {}
    det_file = "dataset/train/%s/%s/gt/gt.txt" % (seq, cam)
    return read_detections(det_file, n_frames)


def load_gt(seq, cameras):
    gt = {}
    for i in range(cameras.shape[0]):
        cam = cameras.iloc[i]
        det_file = "dataset/train/%s/%s/gt/gt.txt" % (seq, cam['Name'])
        gt[cam['Name']] = read_detections(det_file, cam['Frames'])

    return gt
