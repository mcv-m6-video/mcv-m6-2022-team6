import argparse

import cv2
import numpy as np

from Week1 import flow_read, msen, pepn
import time
import pickle

from flow_block_matching import get_optical_flow, METRICS

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

	# img1 = cv2.imread("../data/results_opticalflow_kitti/original_black/000157_10.png")
	# img2 = cv2.imread("../data/results_opticalflow_kitti/original_black/000157_11.png")
	# F_gt = flow_read("../data/results_opticalflow_kitti/groundtruth/000157_10.png")

	mode = ["backward", "forward"]
	method = ["full", "cv2", "log"]
	block_size = [4, 8, 16, 32]
	area = [4, 8, 16, 32]
	log_step = [2, 4, 8]
	metric = {
		"full": ["SAD", "SSD", "MAE", "MSE"],
		"cv2": ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
				'cv2.TM_CCORR_NORMED'],
		"log": ["Only"]
	};

	results = [];


	def execute_optical(img1, img2, bs, a, m, met, m_fn, ls=4):
		try:
			start_time = time.time()
			print("---------------------------------")
			print(bs, a, m, met, m_fn, ls);
			result = get_optical_flow(img1, img2, bs, a, m, met, ls, m_fn);
			MSEN = msen(F_gt, result)
			PEPN = pepn(F_gt, result, 3)
			print('MSEN:', MSEN)
			print('PEPN:', PEPN)
			print("--- %s seconds ---" % (time.time() - start_time))
			results.append([bs, a, m, met, m_fn, ls, MSEN, PEPN, time.time() - start_time])
		except:
			print("Error under execution")


	for m in mode:
		for met in method:
			for bs in block_size:
				for a in area:
					for m_fn in metric[met]:
						if method == "log":
							for ls in log_step:
								execute_optical(img1, img2, bs, a, m, met, m_fn, ls)
						else:
							execute_optical(img1, img2, bs, a, m, met, m_fn)

	np.save('results/results_task1_1.npy', np.array(results))
