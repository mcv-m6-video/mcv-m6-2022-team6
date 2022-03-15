import utils as ut
import voc_evaluation
import numpy as np
import os

os.chdir("..")
# PATHS
annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'
detections_path = {
	'rcnn': 'data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
	'ssd': 'data/AICity_data/train/S03/c010/det/det_ssd512.txt',
	'yolo': 'data/AICity_data/train/S03/c010/det/det_yolo3.txt'
}

if __name__ == '__main__':

	frame_id = 1500
	display = True
	test_det = True

	annotations = ut.read_annotations(annotations_path)

	if test_det:
		detections = ut.annotations_to_detections(annotations) #Test
	else:
		detections = ut.read_detections(detections_path['rcnn']) #Real

	if display:
		frameA= ut.read_frame_bw(video_path, frame_id)
		frameB= ut.read_frame_bw(video_path, frame_id+1)
		for i in range(np.shape(frameA)[1]):
			for j in range(np.shape(frameA)[2]):
				A = frameA[i,j]
#u = (A^t A)^-1 A^T B
		frame = frameB-frameA
		ut.display_frame(frame)

	#One frame
	iou_frame = ut.get_frame_iou(annotations[frame_id], detections[frame_id]);
	print(iou_frame)
	#rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)

	# Overall detections
	rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)
	print(rec)
	print(prec)
	print(ap)
