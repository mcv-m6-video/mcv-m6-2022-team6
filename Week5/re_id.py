import os, sys, argparse
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.distances import CosineSimilarity

from re_id_utils import get_id_cam, get_data, comp_cams, merge, invert_dict, load_siamese
from eval_tracking_mtmc import evaluate_mtmc

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_csv', type=str, default='Week5/det/det_car_patches_annotations.csv',
                        help='path to det csv containing annotations')
    parser.add_argument('--det_patches', type=str, default='Week5/det/det_patches/',
                        help='path to det folder containing car patches')
    parser.add_argument('--trunk_model', type=str, default='Week5/model/0001_32_256/trunk.pth',
                        help='path to trunk model')
    parser.add_argument('--embedder_model', type=str, default='Week5/model/0001_32_256/embedder.pth',
                        help='path to embedder model')
    parser.add_argument('--thr', type=float, default=0.5,
                        help='threshold to consider a match')
    parser.add_argument('--patches_to_compare_c1', type=int, default=4,
                        help='number of patches to compare with cam1')
    parser.add_argument('--patches_to_compare_c2', type=int, default=6,
                        help='number of patches to compare with cam2')
    parser.add_argument('--show_reid', default=True,
                        help='show example of reid')
    parser.add_argument('--save_reid', type=str, default=None,#'Week5/det/reid/',
                        help='path to save reid detections')
    parser.add_argument('--eval_mtmc', default=True,
                        help='evaluate multi target multi camera tracking')
    
    
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    backbone, encoder = load_siamese(args.trunk_model, args.embedder_model)

    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=args.thr)
    inference_model = InferenceModel(backbone, encoder, match_finder=match_finder)

    IDs = pd.read_csv(args.det_csv)
    data, _ = get_data('test', IDs, args.det_patches)
    idx_cameras = c_f.get_labels_to_indices(data.camera)  #a dictionary with each camera (c010,c011,...) as a key and an array with frame_idx as values

    id_frames = {}
    for cam in ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']:
        id_frames[cam] = get_id_cam(data, idx_cameras, cam) #get for each camera box_ids and frame_idx

    id_frames_ref = deepcopy(id_frames['c010'])

    for cam in tqdm(['c011', 'c012', 'c013', 'c014', 'c015']):
        reID_cam = comp_cams(data, deepcopy(id_frames_ref),
                                 deepcopy(id_frames[cam]), inference_model,
                                 args.patches_comp_c1,
                                 args.patches_comp_c2)

        id_frames_ref = merge(id_frames_ref, reID_cam)

    frames_id_all = invert_dict(id_frames_ref)

    det_dir = os.path.join(args.save_reid,
                            ''.join(str(args.thr).split('.')) + '_' + str(args.patches_comp_c1) + '_' + str(args.patches_comp_c2))

    if args.save_reid is not None:
        for cam, id_frames_cam in id_frames.items():
            os.makedirs(os.path.join(det_dir, cam), exist_ok=True)
            det_path = os.path.join(det_dir, cam, 'overlap_reid_detections.txt')
            if os.path.isfile(det_path):
                os.remove(det_path)

            frames_cam = []
            for tmp_frames in list(id_frames_cam.values()):
                frames_cam.extend(tmp_frames)
            frames_cam = sorted(frames_cam)

            for idx in frames_cam:
                box = data[idx]

                # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                det = str(box[4] + 1) + ',' + str(frames_id_all[idx]) + ',' + str(box[3][0]) + ',' \
                      + str(box[3][1]) + ',' + str(box[3][2]) + ',' + str(box[3][3]) + ',' + '1' + ',-1,-1,-1\n'

                with open(det_path, 'a+') as f:
                    f.write(det)

    if args.eval_mtmc:
        summary = evaluate_mtmc(det_dir,"dataset/train/S03")
        print('thr: ', args.thr, ', patches c1: ', args.patches_comp_c1, ', patches c2: ', args.patches_comp_c2)
        print(summary)

        summary.to_csv(os.path.join(det_dir, 'evaluation_mtmc.csv'), sep='\t')
