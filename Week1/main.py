import utils_week1 as ut
import voc_evaluation_week1 as voc_evaluation
import matplotlib.pyplot as plt
import numpy as np
# PATHS
annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'
detections_path = {
	'rcnn': 'data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
	'ssd': 'data/AICity_data/train/S03/c010/det/det_ssd512.txt',
	'yolo': 'data/AICity_data/train/S03/c010/det/det_yolo3.txt'
}
predictions_from = 'yolo'

if __name__ == '__main__':

	frame_id = 1500
	display = False
	test_det = False

	annotations = ut.read_annotations(annotations_path)

	if test_det:
		detections = ut.annotations_to_detections(annotations) #Test
	else:
		detections = ut.read_detections(detections_path[predictions_from]) #Real

	if display:
		ut.display_frame(ut.read_frame(video_path, frame_id), annotations[frame_id], detections[frame_id])

	#One frame
	iou_frame = ut.get_frame_iou(annotations[frame_id], detections[frame_id]);
	print(iou_frame)
	#rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)

	# Overall detections
	rec, prec, ap = voc_evaluation.voc_eval(annotations, detections,use_confidence=True)
	print('recall from {} = {}'.format(predictions_from,rec))
	print('precision from {} = {}'.format(predictions_from,prec))
	print('ap from {} = {}'.format(predictions_from,ap))

