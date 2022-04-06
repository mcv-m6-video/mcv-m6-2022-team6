import argparse
import sys
import cv2

sys.path.append('./')
from Week1 import flow_read, msen, pepn, plotFlow
import time
import numpy as np
from matplotlib import pyplot as plt

from flow_block_matching import get_optical_flow

TOTAL_FRAMES = 2141

def task4_1_1(block_size=32):
	parser = argparse.ArgumentParser(description="Optical flow")
	parser.add_argument('--preview', default=False)
	parser.add_argument('--mode', default='backward')
	args = parser.parse_args()

	#load images 45 or 157
	image_number=45
	
	img_path = "data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number)
	img1 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number))
	img2 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_11.png".format(image_number))
	F_gt = flow_read("data/results_opticalflow_kitti/groundtruth/{:0>6}_10.png".format(image_number))

	start_time = time.time()
	result = get_optical_flow(img1, img2, mode=args.mode, method="cv2", show_preview=True,block_size=block_size);
	print("--- %s seconds ---" % (time.time() - start_time))

	MSEN = msen(F_gt, result)
	PEPN = pepn(F_gt, result, 3)
	print('MSEN:', MSEN)
	print('PEPN:', PEPN)

	
	plotFlow(result, img_path);

	cv2.waitKey(0)
	


def task4_1_1_method():
	parser = argparse.ArgumentParser(description="Optical flow")
	parser.add_argument('--preview', default=False)
	parser.add_argument('--mode', default='backward')
	args = parser.parse_args()

	#load images 45 or 157
	image_number=45
	
	img_path = "data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number)
	img1 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number))
	img2 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_11.png".format(image_number))
	F_gt = flow_read("data/results_opticalflow_kitti/groundtruth/{:0>6}_10.png".format(image_number))
	MSEN_array = []
	PEPN_array = []
	flow_time_array = []
	method_array = ["cv2CCOEFF", "cv2CCORR","cv2SQDIFF","log","full"]
	b_size = 32
	for method_i in method_array:
		start_time = time.time()
		result = get_optical_flow(img1, img2, mode=args.mode, method=method_i, show_preview=True,block_size=b_size);
		print("With area size of"+ str(b_size))
		flow_time = time.time() - start_time
		print("--- %s seconds ---" % (flow_time))

		MSEN = msen(F_gt, result, method_i)
		PEPN = pepn(F_gt, result, 3)
		print('MSEN:', MSEN)
		print('PEPN:', PEPN)

		flow_time_array.append(flow_time)
		MSEN_array.append(MSEN)
		PEPN_array.append(PEPN)
		plotFlow(result, img_path);
	MSEN_PEPN_plot(method_array, flow_time_array, MSEN_array, PEPN_array, "methods")
	
	print (MSEN_array)
	cv2.waitKey(0)

def task4_1_1_area():
	parser = argparse.ArgumentParser(description="Optical flow")
	parser.add_argument('--preview', default=False)
	parser.add_argument('--mode', default='backward')
	args = parser.parse_args()

	#load images 45 or 157
	image_number=45
	MSEN_array = []
	PEPN_array = []
	flow_time_array = []
	img_path = "data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number)
	img1 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number))
	img2 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_11.png".format(image_number))
	F_gt = flow_read("data/results_opticalflow_kitti/groundtruth/{:0>6}_10.png".format(image_number))
	MSEN_array = []
	area_size_to_check = [4, 8, 16, 32, 64, 128, 256]
	for b_size in area_size_to_check:
		start_time = time.time()
		result = get_optical_flow(img1, img2, mode=args.mode, method="log", show_preview=True,block_size=b_size);
		print("With area size of"+ str(b_size))
		flow_time = time.time() - start_time
		print("--- %s seconds ---" % (flow_time))

		MSEN = msen(F_gt, result,"BlockSize"+ str(b_size))
		PEPN = pepn(F_gt, result, 3)
		flow_time_array.append(flow_time)
		MSEN_array.append(MSEN)
		PEPN_array.append(PEPN)
		print('MSEN:', MSEN)
		print('PEPN:', PEPN)
		plotFlow(result, img_path)

	MSEN_PEPN_plot(area_size_to_check, flow_time_array, MSEN_array, PEPN_array, "areaBlock")

	print (MSEN_array)
	cv2.waitKey(0)

def task4_1_1_step():
	parser = argparse.ArgumentParser(description="Optical flow")
	parser.add_argument('--preview', default=False)
	parser.add_argument('--mode', default='backward')
	args = parser.parse_args()

	#load images 45 or 157
	image_number=45
	MSEN_array = []
	PEPN_array = []
	flow_time_array = []
	img_path = "data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number)
	img1 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number))
	img2 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_11.png".format(image_number))
	F_gt = flow_read("data/results_opticalflow_kitti/groundtruth/{:0>6}_10.png".format(image_number))
	MSEN_array = []
	step_array = [1, 2, 4,6, 8]
	for step in step_array:
		start_time = time.time()
		result = get_optical_flow(img1, img2, mode=args.mode, method="log", show_preview=True,block_size=32, log_step=step);
		print("With area size of"+ str(step))
		flow_time = time.time() - start_time
		print("--- %s seconds ---" % (flow_time))

		MSEN = msen(F_gt, result, "step"+str(step))
		PEPN = pepn(F_gt, result, 3)
		flow_time_array.append(flow_time)
		MSEN_array.append(MSEN)
		PEPN_array.append(PEPN)
		print('MSEN:', MSEN)
		print('PEPN:', PEPN)
		plotFlow(result, img_path)

	MSEN_PEPN_plot(step_array, flow_time_array, MSEN_array, PEPN_array, "step_size")

	print (MSEN_array)
	cv2.waitKey(0)

def MSEN_PEPN_plot(area_size_to_check, flow_time_array, MSEN_array,PEPN_array,outputName):
	X = np.arange(len(area_size_to_check))
	fig, ax1 = plt.subplots()

	ax1.set_xlabel(outputName)
	ax1.set_ylabel("MSEN", color= "orange")
	xlabel = ["{}\n {:.3f}s".format(method_array_, flow_time_array_) for flow_time_array_, method_array_ in zip(flow_time_array, area_size_to_check)]
	ax1.plot(xlabel ,MSEN_array, color= "orange")
	ax1.set_ylim([0,40])
	ax1.tick_params(axis='y', labelcolor="orange")

	ax2 = ax1.twinx()
	ax2.set_ylabel("PEPN", color="blue")
	ax2.set_ylim([0,100])
	
	ax2.plot(PEPN_array, color= "blue")
	ax2.tick_params(axis='y', labelcolor="blue")

	fig.tight_layout()
	plt.show()
	plt.savefig(str(outputName)+'.png')

if __name__ == '__main__':
	#task4_1_1_method() 
	# best option log (as faster as cv2 but better performance without fine tunning)
	#task4_1_1_area()
	# best area 32
	task4_1_1_step()