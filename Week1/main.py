import pandas as pd
import utils as ut
import voc_evaluation
import cv2

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
		detections = ut.read_detections(detections_path['ssd']) #Real

	if display:
		frame = ut.read_frame(video_path, frame_id)
		for r in annotations[frame_id]:
			frame = cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)

		for r in detections[frame_id]:
			bbox = r['bbox']
			frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

		imS = cv2.resize(frame, (960, 540))
		cv2.imshow('Frame', imS)
		cv2.waitKey(0)  # waits until a key is pressed
		cv2.destroyAllWindows()

	#One frame
	iou_frame = ut.get_frame_iou(annotations[frame_id], detections[frame_id]);
	print(iou_frame)
	#rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)

	# Overall detections
	rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)
	print(rec)
	print(prec)
	print(ap)
