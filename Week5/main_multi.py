import argparse

import numpy as np
import pandas as pd
import cv2

FPS = 10


def get_frame_id(frame, timestamp):
	real_frame = int(frame - np.ceil(timestamp * FPS))
	print("Frame: %04d Real: %04d" % (frame, real_frame))
	return real_frame


def get_max_frame(cameras):
	last_camera = cameras['Time'].argmax()
	camera = cameras.iloc[last_camera]
	return get_frame_id(camera['Frames'], camera['Time'])


def get_cameras_info(sequence, cams):
	cameras = pd.read_csv('dataset/cam_framenum/%s.txt' % sequence, delimiter=' ', header=None,
						  names=['Cam', 'Frames'])
	ncams = cameras.shape[0];

	cameras['Time'] = pd.read_csv('dataset/cam_timestamp/%s.txt' % sequence, delimiter=' ',
								  header=None, names=['Time'], usecols=[1])

	cameras['Start'] = [get_frame_id(0, cameras['Time'][i]) for i in range(ncams)]
	cameras['End'] = [int(cameras['Start'][i] + cameras['Frames'][i]) for i in range(ncams)]
	cameras = cameras.loc[cameras['Cam'].isin(cams)];

	max_frame = get_max_frame(cameras)

	return cameras, max_frame


def read_frame(sequence, frameId, camera):
	print("Camera %s" % camera['Cam'])
	img = cv2.imread("dataset/train/%s/%s/img/%04d.jpeg"
					 % (sequence, camera['Cam'], get_frame_id(frameId, camera['Time'])))

	if img is None:
		return np.zeros((1920, 1080, 3))
	else:
		return img


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Multicamera Tracking")
	parser.add_argument('--seq', default='S01')
	parser.add_argument('--cams', default=['c001', 'c002'])
	parser.add_argument('--start_cam', default='c001')
	args = parser.parse_args()

	cameras, max_frame = get_cameras_info(args.seq, args.cams)
	preview_size = (1920 // 3, 1080 // 3)

	for i in range(0, max_frame):
		print("Frame %d" % i);
		img = read_frame(args.seq, i, cameras.iloc[0])
		img2 = read_frame(args.seq, i, cameras.iloc[1])

		cv2.imshow('Cam01', cv2.resize(img, preview_size))
		cv2.imshow('Cam02', cv2.resize(img2, preview_size))

		cv2.waitKey(0);

	"""max_frame = get_max_frame(cam_frames, cam_timestamp);

	for i in range(0, cam_frames['Frames'][args.start_cam]):
		print(i)"""
