import utils as ut
import voc_evaluation
import numpy as np
import os
import cv2 as cv

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

		
	F_gt = ut.flow_read("C:/Users/eudal/Documents/M6/mcv-m6-2022-team6-main/data/results_opticalflow_kitti/groundtruth/000045_10.png")
	F_test = ut.flow_read("C:/Users/eudal/Documents/M6/mcv-m6-2022-team6-main/data/results_opticalflow_kitti/results/LKflow_000045_10.png")

	MSEN = ut.msen(F_gt, F_test)
	
	print(MSEN)

	input()
