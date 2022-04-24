import argparse

import cv2

from Week5.deepsort import deepsort_rbc
from utils import read_frame
from tracking import TrackingKalmanSort
from detections import load_single_detections, load_gt_single
from cameras import get_cameras_info, get_frame_id
from plot import plot_camera
import numpy as np

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Multicamera Tracking")
	parser.add_argument('--seq', default='S03')
	parser.add_argument('--cam', default='c010')
	parser.add_argument('--det', default='mask_rcnn')
	parser.add_argument('--preview', default=True)
	args = parser.parse_args()

	cameras, _ = get_cameras_info(args.seq, [args.cam])
	max_frame = cameras.iloc[0]['Frames']
	detections = np.array(load_single_detections(args.seq, args.cam, args.det, max_frame))
	gt = load_gt_single(args.seq, args.cam, max_frame)

	# Init trackers
	deepsort = deepsort_rbc(wt_path='ckpts/model640.pt')

	for frameId in range(1, max_frame):

		# Load frame
		frame = cv2.imread("dataset/train/%s/%s/img/%04d.jpeg" % (args.seq, args.cam, frameId))

		# Get detections
		detec = detections[frameId - 1][:, :4]
		out_scores = detections[frameId - 1][:, 4]

		# Process tracking
		tracker, detections_class = deepsort.run_deep_sort(frame, out_scores, detections)

		# Plot cameras
		if bool(args.preview):
			plot_camera(args.cam, frame, detec)

		cv2.waitKey(10)
