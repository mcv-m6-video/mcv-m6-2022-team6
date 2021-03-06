import argparse

import cv2
from Week1 import flow_read, msen, pepn, plotFlow
import time

from flow_block_matching import get_optical_flow

TOTAL_FRAMES = 2141

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Optical flow")
	parser.add_argument('--preview', default=False)
	parser.add_argument('--mode', default='backward')
	parser.add_argument('--method', default='full')
	args = parser.parse_args()

	img_path = "../data/results_opticalflow_kitti/original/000045_10.png"
	img1 = cv2.imread("../data/results_opticalflow_kitti/original/000045_10.png")
	img2 = cv2.imread("../data/results_opticalflow_kitti/original/000045_11.png")
	F_gt = flow_read("../data/results_opticalflow_kitti/groundtruth/000045_10.png")


	#img1 = cv2.imread("../data/results_opticalflow_kitti/original_black/000157_10.png")
	#img2 = cv2.imread("../data/results_opticalflow_kitti/original_black/000157_11.png")
	#F_gt = flow_read("../data/results_opticalflow_kitti/groundtruth/000157_10.png")

	start_time = time.time()
	result = get_optical_flow(img1, img2, mode=args.mode, method=args.method, show_preview=args.preview);
	print("--- %s seconds ---" % (time.time() - start_time))

	MSEN = msen(F_gt, result)
	PEPN = pepn(F_gt, result, 3)
	print('MSEN:', MSEN)
	print('PEPN:', PEPN)

	plotFlow(result, img_path);