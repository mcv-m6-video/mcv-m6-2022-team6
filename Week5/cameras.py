import pandas as pd
import numpy as np

FPS = 10.0


def get_frame_id(frame, timestamp):
	real_frame = int(frame - np.around(timestamp * FPS))
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
	cameras = cameras.loc[cameras['Cam'].isin(cams)]

	max_frame = get_max_frame(cameras)

	return cameras, max_frame
