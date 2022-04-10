import cv2
import numpy as np

from Week5 import get_frame_id


def read_frame(sequence, frame_id, camera):
	real_frame_id = get_frame_id(frame_id, camera['Time'])
	if real_frame_id <= 0:
		return np.zeros((1920, 1080, 3))

	return cv2.imread("dataset/train/%s/%s/img/%04d.jpeg"
					  % (sequence, camera['Name'], real_frame_id))


def get_rect_iou(a, b):  # intersection over union
	"""Return iou for a pair of boxes"""
	x11, y11, x12, y12 = a
	x21, y21, x22, y22 = b

	xA = max(x11, x21)
	yA = max(y11, y21)
	xB = min(x12, x22)
	yB = min(y12, y22)

	# respective area of ​​the two boxes
	boxAArea = (x12 - x11) * (y12 - y11)
	boxBArea = (x22 - x21) * (y22 - y21)

	# overlap area
	interArea = max(xB - xA, 0) * max(yB - yA, 0)

	# IOU
	return interArea / (boxAArea + boxBArea - interArea)


def get_rect_ioa(a, b):  # intersection over areas
	"""Return ioa for a pair of boxes"""
	x11, y11, x12, y12 = a
	x21, y21, x22, y22 = b

	xA = max(x11, x21)
	yA = max(y11, y21)
	xB = min(x12, x22)
	yB = min(y12, y22)

	# respective area of ​​the two boxes
	boxAArea = (x12 - x11) * (y12 - y11)
	boxBArea = (x22 - x21) * (y22 - y21)

	# overlap area
	interArea = max(xB - xA, 0) * max(yB - yA, 0)

	# IOA
	return interArea / boxAArea, interArea / boxBArea
