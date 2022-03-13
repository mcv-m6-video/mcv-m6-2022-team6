import pandas as pd
import utils as ut
import voc_evaluation

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

	print(voc_evaluation.voc_eval(annotations, detections))
