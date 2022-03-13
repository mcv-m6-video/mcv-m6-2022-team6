import pandas as pd
import utils as ut
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# PATHS
annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'
detections_path = {
	'rcnn': 'data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
	'ssd': 'data/AICity_data/train/S03/c010/det/det_ssd512.txt',
	'yolo': 'data/AICity_data/train/S03/c010/det/det_yolo3.txt'
}

if __name__ == '__main__':
	annotations = ut.read_annotations(annotations_path)
	detections = ut.read_detections(detections_path['rcnn'])

	evaluator = COCOEvaluator("balloon_val", output_dir="./output")
	val_loader = build_detection_test_loader(cfg, "balloon_val")
	print(inference_on_dataset(predictor.model, val_loader, evaluator))