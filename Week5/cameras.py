import pandas as pd
import numpy as np

FPS = 10.0


def get_frame_id(frame, timestamp):
	real_frame = int(frame - np.around(timestamp * FPS))
	return real_frame


def get_max_frame(cameras):
	last_camera = cameras['Time'].argmax()
	camera = cameras.iloc[last_camera]
	return get_frame_id(camera['Frames'], camera['Time'])


def get_cameras_info(sequence, cams):
	cameras = pd.read_csv('dataset/cam_framenum/%s.txt' % sequence, delimiter=' ', header=None,
						  names=['Name', 'Frames'])
	ncams = cameras.shape[0];

	cameras['Time'] = pd.read_csv('dataset/cam_timestamp/%s.txt' % sequence, delimiter=' ',
								  header=None, names=['Time'], usecols=[1])

	cameras = cameras.loc[cameras['Name'].isin(cams)]

	max_frame = get_max_frame(cameras)

	return cameras, max_frame
