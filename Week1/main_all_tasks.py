import utils_week1 as ut
import voc_evaluation
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

def task101():
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
		detections = ut.read_detections(detections_path[predictions_from]) #Real

	if display:
		frame = ut.print_bboxes(ut.read_frame(video_path, frame_id), annotations[frame_id], detections[frame_id])
		ut.display_frame(frame)

	#One frame
	iou_frame = ut.get_frame_iou(annotations[frame_id], detections[frame_id]);	
	#rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)

	# Overall detections
	rec, prec, ap = voc_evaluation.voc_eval(annotations, detections,use_confidence=True)
	print('recall from {} = {}'.format(predictions_from,rec))
	print('precision from {} = {}'.format(predictions_from,prec))
	print('ap from {} = {}'.format(predictions_from,ap))


def task102():
    display = False
    test_det = False
    generate_noise = True
    noisy_percent = 0.5
    compare_detections_plot = False

    annotations = ut.read_annotations(annotations_path)

    if test_det:
        detections = ut.annotations_to_detections(
            annotations, generate_noise, noisy_percent)
    else:
        detections = ut.read_detections(detections_path[predictions_from]) #Real
    if compare_detections_plot:
        ut.plot_multiple_IoU(detections_path,annotations)
    score_for_each_frame = []
    
    for frame_id in range(len(annotations)):
        score_for_each_frame.append(ut.get_frame_iou(
            annotations[frame_id], detections[frame_id]))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(0, len(annotations), 1), score_for_each_frame)
    ax.set(xlabel='frame', ylabel='IoU')
    ax.set_ylim(0, 1)
    ax.set_xlim(0,len(annotations))
    plt.savefig('task2_plot.png')

    #rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)

    # Overall detections
    rec, prec, ap = voc_evaluation.voc_eval(annotations, detections, use_confidence=True)
    print('recall from {} = {}'.format(predictions_from,rec))
    print('precision from {} = {}'.format(predictions_from,prec))
    print('ap from {} = {}'.format(predictions_from,ap))
    print('mean IoU {}'.format(np.mean(score_for_each_frame)))

def task103():
	F_gt = ut.flow_read("data/results_opticalflow_kitti/groundtruth/000045_10.png")
	F_test = ut.flow_read("data/results_opticalflow_kitti/results/LKflow_000045_10.png")

	MSEN = ut.msen(F_gt, F_test)
	PEPN = ut.pepn(F_gt, F_test, 3)
	
	print('MSEN:', MSEN)
	print('PEPN:',PEPN)

def task104():
	F_gt = ut.flow_read("data/results_opticalflow_kitti/groundtruth/000045_10.png")
	img_path = "data/results_opticalflow_kitti/original_black/000045_10.png"
	ut.plotFlow(F_gt, img_path)


if __name__ == '__main__':
    task101()
    task102()
    task103()
    task104()