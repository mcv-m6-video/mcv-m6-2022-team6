import argparse

import numpy as np

import cv2

from Week5.cameras import get_frame_id, get_cameras_info


def read_frame(sequence, frameId, camera):
	real_frame_id = get_frame_id(frameId, camera['Time'])
	if real_frame_id <= 0:
		return np.zeros((1920, 1080, 3))

	print("Camera %s" % camera['Cam'])
	return cv2.imread("dataset/train/%s/%s/img/%04d.jpeg"
					  % (sequence, camera['Cam'], real_frame_id))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Multicamera Tracking")
	parser.add_argument('--seq', default='S03')
	parser.add_argument('--cams', default=['c003', 'c004'])
	parser.add_argument('--start_cam', default='c001')
	args = parser.parse_args()

	cameras, max_frame = get_cameras_info(args.seq, args.cams)
	preview_size = (1920 // 3, 1080 // 3)

	for i in range(1, max_frame):
		print("Frame %d" % i)
		img = read_frame(args.seq, i, cameras.iloc[0])
		img2 = read_frame(args.seq, i, cameras.iloc[1])

		cv2.imshow('Cam01', cv2.resize(img, preview_size))
		cv2.imshow('Cam02', cv2.resize(img2, preview_size))

		cv2.waitKey(0)
