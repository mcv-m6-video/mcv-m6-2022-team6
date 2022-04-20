import os, sys, cv2, argparse
from tqdm import tqdm

sys.path.append('Week3')
from utils_week3 import  read_detections, read_annotations, read_detections1, read_detections2

import motmetrics as mm


def filter_bboxes_size(det_bboxes):
    filtered_bboxes = []
    for b in det_bboxes:
        if (b['bbox'][2]-b['bbox'][0]) >= 70 and (b['bbox'][3]-b['bbox'][1]) >= 56:
            filtered_bboxes.append(b)

    return filtered_bboxes


def eval_tracking(num_frames, gt, det, accumulator, cam):
    print(num_frames)
    for frame_id in tqdm(range(num_frames), desc='Evaluating Cam ' + cam):
        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]

        det_bboxes = []
        if frame_id in det:
            det_bboxes = filter_bboxes_size(det[frame_id])

        objs =[]
        hyps=[]
        for bbox in gt_bboxes:
            center = [(bbox['bbox'][0] + bbox['bbox'][2]) // 2, (bbox['bbox'][1] + bbox['bbox'][3]) // 2]
            objs.append(center)
        for bbox in det_bboxes:
            center1 = [(bbox['bbox'][0] + bbox['bbox'][2]) // 2, (bbox['bbox'][1] + bbox['bbox'][3]) // 2]
            hyps.append(center1)

        


        accumulator.update(
            [bbox['id'] for bbox in gt_bboxes],  # Ground truth objects in this frame
            [bbox['id'] for bbox in det_bboxes],  # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(objs, hyps)
            # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )

    return accumulator


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="dataset/train",
                        help='path to sequences of AICity')

    parser.add_argument('--reid_path', type=str, default="Week5/det/reid/06_3_5",
                        help='path to reid')

    parser.add_argument('--seq', type=str, default='S03',
                        help='sequence/camera from AICity dataset')

    return parser.parse_args(args)


def evaluate_mtmc(reid_path, data_path):
    accumulator = mm.MOTAccumulator(auto_id=True)
    for cam in sorted(os.listdir(data_path)):

        vidcap = cv2.VideoCapture(os.path.join(data_path, cam, 'vdo.avi'))
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        gt = read_detections2(os.path.join(data_path, cam, 'gt/gt.txt'))
        det_reid = read_detections2(os.path.join(reid_path, cam, 'overlap_reid_detections.txt'))

        accumulator = eval_tracking(num_frames, gt, det_reid, accumulator, cam)

    mh = mm.metrics.create()
    return mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')


if __name__ == "__main__":
    args = parse_args()
    cams_path = os.path.join(args.reid_path)

    print('[INFO] Evaluating MTMC tracking, sequence ', args.seq)
    accumulator = mm.MOTAccumulator(auto_id=True)
    for cam in sorted(os.listdir(cams_path)):

        vidcap = cv2.VideoCapture(os.path.join(args.data_path, args.seq, cam, 'vdo.avi'))
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        gt = read_detections2(os.path.join(args.data_path, args.seq, cam, 'gt/gt.txt'))
        det_reid = read_detections2(os.path.join(cams_path, cam, 'overlap_reid_detections.txt'))

        accumulator = eval_tracking(num_frames, gt, det_reid, accumulator, cam)

    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
    print(summary)