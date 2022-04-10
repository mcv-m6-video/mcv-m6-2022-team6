import argparse

import cv2

from utils import read_frame
from tracking import TrackingKalmanSort
from detections import load_detections, load_gt
from cameras import get_cameras_info, get_frame_id
from plot import plot_camera

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Multicamera Tracking")
	parser.add_argument('--seq', default='S03')
	parser.add_argument('--cams', default=['c010', 'c011'])
	parser.add_argument('--start_cam', default='c010')
	parser.add_argument('--det', default='mask_rcnn')
	args = parser.parse_args()

	cameras, max_frame = get_cameras_info(args.seq, args.cams)
	detections = load_detections(args.seq, cameras, args.det)
	gt = load_gt(args.seq, cameras)
	#tracker = TrackingKalmanSort()

	for i in range(1, max_frame):

		for cam_i in cameras.index:
			# Camera data
			cam = cameras.iloc[cam_i]
			frameId = get_frame_id(i, cam['Time'])

			# Load frame
			frame = read_frame(args.seq, i, cam)

			detec = []
			if frameId > 0:
				detec = detections[cam['Name']][frameId - 1]

			plot_camera(cam['Name'], frame, detec)

		cv2.waitKey(10)
