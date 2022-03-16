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

def task103():
	F_gt = ut.flow_read("data/results_opticalflow_kitti/groundtruth/000045_10.png")
	F_test = ut.flow_read("data/results_opticalflow_kitti/results/LKflow_000045_10.png")

	MSEN = ut.msen(F_gt, F_test)
	PEPN = ut.pepn(F_gt, F_test, 0.5)
	
	print(MSEN)
	print(PEPN)

def task104():
	F_gt = ut.flow_read("data/results_opticalflow_kitti/groundtruth/000045_10.png")
	ut.plotFlow(F_gt)

if __name__ == '__main__':

	frame_id = 1500
	display = False
	test_det = False
	generate_noise = False
	noisy_percent = 1
	dropout = False
	dropout_percent = 0.90

	annotations = ut.read_annotations(annotations_path)

	if test_det:
		detections = ut.annotations_to_detections(annotations, generate_noise, noisy_percent, dropout, dropout_percent) #Test
	else:
		detections = ut.read_detections(detections_path['yolo']) #Real

	if display:
		frame = ut.print_bboxes(ut.read_frame(video_path, frame_id), annotations[frame_id], detections[frame_id])
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
