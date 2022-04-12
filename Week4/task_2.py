import os, sys, cv2, argparse
import numpy as np
from tqdm import tqdm
import motmetrics as mm
from collections import defaultdict, OrderedDict
import imageio
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("Week1")
#from utils_week1 import read_detections

sys.path.append("Week3")
from Tracking import TrackingBase, TrackingIOU, TrackingKalman, TrackingKalmanSort, TrackingIOUDirection
#from Task2 import draw_bbox
from sort import Sort
from utils_week3 import read_detections, read_annotations, read_detections1, read_detections2, draw_boxes, draw_boxes1

det_filename = {
    'retina': 'detections.txt',
    'mask': 'det_mask_rcnn.txt',
    'ssd': 'det_ssd512.txt',
    'yolo': 'det_yolo3.txt'
}

#annotations_path= 'data/ai_challenge_s03_c010-full_annotation.xml'


def filter_bboxes_size(det_bboxes):
    filtered_bboxes = []
    for b in det_bboxes:
        if (b['bbox'][2]-b['bbox'][0]) >= 70 and (b['bbox'][3]-b['bbox'][1]) >= 56:
            filtered_bboxes.append(b)

    return filtered_bboxes


def filter_bboxes_parked(list_positions, list_positions_bboxes, var_thr):
    det_bboxes_filtered = []
    for index, centers_bboxes_same_id in list_positions.items():
        id_var = np.mean(np.std(centers_bboxes_same_id, axis=0))
        if id_var > var_thr:
            for b in list_positions_bboxes[index]:
                det_bboxes_filtered.append(b)

    return det_bboxes_filtered


def group_by_frame(boxes):
    grouped = defaultdict(list)
    for box in boxes:
        grouped[box['frame']].append(box)
    return OrderedDict(sorted(grouped.items()))
    
#writer = imageio.get_writer("Week4/tracking2.gif", mode="I", fps=30) #gift
def filter_detections_parked(params, mse_thr=300, var_thr=50):
    print("[INFO] Filtering detections to remove parked cars")

    vidcap = cv2.VideoCapture(params['video_path'])
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt_ids, gt = read_detections1(params["gt_path"], num_frames)
    det = read_detections(params["det_path"], iou_th=0.5)

    #gt_ids, annotations = read_annotations(annotations_path, True)

    detections = []
    list_positions = {}
    list_positions_bboxes = {}

    center_seen_last5frames = {}
    id_seen_last5frames = {}

    if params['track_method'] == 'overlap':
        tracker = TrackingIOU(gt_ids, gt)
    elif params['track_method'] == 'kalman':
        tracker = TrackingKalmanSort(gt_ids, gt)

    det_bboxes_old = -1

    esc_pressed = False

    for frame_id in tqdm(range(num_frames)):
        _, frame = vidcap.read()

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]

        det_bboxes = []
        if frame_id in det:
            if params['track_method'] == 'overlap':
                det_bboxes = tracker.generate_track(frame_id,det[frame_id])
            elif params['track_method'] == 'kalman':
                det_bboxes = tracker.generate_track(frame_id, det[frame_id])

        detections += det_bboxes

        id_seen = []

        for object_bb in det_bboxes:
            center=[(object_bb['bbox'][0] + object_bb['bbox'][2]) // 2, (object_bb['bbox'][1] + object_bb['bbox'][3]) // 2]
            if object_bb['id'] in list(list_positions.keys()):
                if frame_id < 5:
                    id_seen_last5frames[object_bb['id']] = object_bb['id']
                    center_seen_last5frames[object_bb['id']] = center

                list_positions[object_bb['id']].append([int(x) for x in center])
                list_positions_bboxes[object_bb['id']].append(object_bb)
            else:
                if frame_id < 5:
                    id_seen_last5frames[object_bb['id']] = object_bb['id']
                    center_seen_last5frames[object_bb['id']] = center

                id_seen.append(object_bb)
                list_positions[object_bb['id']] = [[int(x) for x in center]]
                list_positions_bboxes[object_bb['id']] = [object_bb]

        # To detect parked cars
        for bbox in id_seen:
            center1=[(bbox['bbox'][0] + bbox['bbox'][2]) // 2, (bbox['bbox'][1] + bbox['bbox'][3]) // 2]
            for idx in list(id_seen_last5frames.keys()):
                if idx != bbox['id']:
                    center = [center_seen_last5frames[idx]]
                    mse = (np.square(np.subtract(np.array(center), np.array([int(x) for x in center1])))).mean()
                    if mse < mse_thr:
                        bbox['id']= idx

        if params['show_boxes'] and not esc_pressed:
            
            frame = draw_boxes(image=frame, boxes=gt_bboxes, color='g', linewidth=3, boxIds=False,
                               tracker=list_positions)
            frame = draw_boxes(image=frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True,
                               tracker=list_positions)
            cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0))
            cv2.imshow('Frame', frame)
            pressed_key = cv2.waitKey(30)
            if pressed_key == 113:  # press q to quit
                esc_pressed = True
                cv2.destroyAllWindows()
            elif pressed_key == 27:
                sys.exit()
            cv2.resize(frame, (480,270))
            #writer.append_data(frame.astype(np.uint8))
        det_bboxes_old = det_bboxes

    det_bboxes_filtered = filter_bboxes_parked(list_positions, list_positions_bboxes, var_thr)

    return group_by_frame(det_bboxes_filtered)


def eval_tracking(params, det):
    print("Evaluating Tracking")
    
    vidcap = cv2.VideoCapture(params['video_path'])
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt = read_detections2(params["gt_path"], iou_th=0.5)

    # Create an accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    esc_pressed = False

    for frame_id in tqdm(range(num_frames)):
        _, frame = vidcap.read()
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

        if params['show_boxes'] and not esc_pressed:
            frame = draw_boxes1(image=frame, boxes=gt_bboxes, color='g', linewidth=3, boxIds=False)
            frame = draw_boxes(image=frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True)
            cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0))
            cv2.imshow('Frame', frame)
            
            pressed_key = cv2.waitKey(30)
            if pressed_key == 113:  # press q to quit
                esc_pressed = True
                cv2.destroyAllWindows()
            elif pressed_key == 27:
                sys.exit()
            #writer.append_data(frame.astype(np.uint8))
        accumulator.update(
            [bbox['id'] for bbox in gt_bboxes],  # Ground truth objects in this frame
            [bbox['id'] for bbox in det_bboxes],  # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(objs, hyps)
            # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )

    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
    print(summary)


def save_detections(det, det_path):
    for frame_id, boxes in det.items():
        for box in boxes:
            # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            width = abs(box['bbox'][2] - box['bbox'][0])
            height = abs(box['bbox'][3] - box['bbox'][1])
            det = str(box['frame'] + 1) + ',' + str(box['id']) + ',' + str(box['bbox'][0]) + ',' + str(box['bbox'][1]) + ',' + str(width) + ',' + str(height) + ',' + str(1) + ',-1,-1,-1\n'

            with open(det_path, 'a+') as f:
                f.write(det)

    return


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--track_method', type=str, default='kalman',
                        choices=['overlap', 'kalman'],
                        help='method used to track cars')

    parser.add_argument('--det_method', type=str, default='retina',
                        choices=['retina', 'mask', 'ssd', 'yolo'],
                        help='load detections obtained with this method')

    parser.add_argument('--det_dir', type=str, default="data/detections",
                        help='path from where to load detections')

    parser.add_argument('--data_path', type=str, default="data/AICity_data/train",
                        help='path to sequences of AICity')

    parser.add_argument('--seqs', type=lambda s: [str(item) for item in s.split(',')], default=['S03/c010'],
                        help='sequence/camera from AICity dataset')

    parser.add_argument('--show_boxes', action='store_true',
                        help='show bounding boxes')

    parser.add_argument('--save_filtered', default=True,
                        help='save filtered detections (without parked cars)')

    return parser.parse_args(args)


def args_to_params(args, seq):
    return {
        'track_method': args.track_method,
        'det_path': os.path.join(args.det_dir, args.det_method, seq, det_filename[args.det_method]),
        'video_path': os.path.join(args.data_path, seq, 'vdo.avi'),
        'gt_path': os.path.join(args.data_path, seq, 'gt/gt.txt'),
        'show_boxes': args.show_boxes,
    }


if __name__ == "__main__":
    args = parse_args()

    for seq in args.seqs:
        print('---------------------------------------------------------')
        print('Seq: ', seq)
        params = args_to_params(args, seq)

        det_filtered = filter_detections_parked(params)

        if args.save_filtered:
            save_path = os.path.join(args.det_dir, args.det_method, seq, args.track_method + '_filtered_detections.txt')
            save_detections(det_filtered, save_path)
            print('Filtered detections saved in ', save_path)

        #root_path = 'data/detections/retina'
        #det_filtered = read_detections2(os.path.join(root_path, seq, 'overlap_filtered_detections.txt'))

        eval_tracking(params, det_filtered)