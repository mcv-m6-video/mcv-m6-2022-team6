import xmltodict
import numpy as np
from torchvision.io import read_video
import cv2

def annotations_to_detections(annotations, noisy=False):

	detections = {}
	for frame, value in annotations.items():
		detections[frame] = []
		for bbox in value:
			detections[frame].append({
				"bbox": bbox,
				"conf": 1,
				"difficult": False
			})

	return detections


def read_frame(video, frame=0):
	frame = read_video(video, frame, frame)[0]
	img = frame.reshape(frame.shape[1:]).numpy();
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

def read_annotations(path):
	with open(path) as f:
		data = xmltodict.parse(f.read())
		tracks = data['annotations']['track']
		nframes = int(data['annotations']['meta']['task']['size'])

	annotations = {}
	for i in range(nframes):
		annotations[i] = [];

	for track in tracks:
		id = track['@id']
		label = track['@label']

		if label != 'car':
			continue

		for box in track['box']:
			frame = int(box['@frame'])
			annotations[frame].append(np.array([
				float(box['@xtl']),
				float(box['@ytl']),
				float(box['@xbr']),
				float(box['@ybr']),
			]))
	return annotations

def read_detections(path, confidenceThr=0.5):
	"""
	Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
	"""
	with open(path) as f:
		lines = f.readlines()

	detections = {}
	for line in lines:
		det = line.split(sep=',')
		if float(det[6]) > confidenceThr:

			frame = int(det[0])
			if frame not in detections:
				detections[frame - 1] = []

			detections[frame - 1].append({
				"bbox": np.array([float(det[2]),
				float(det[3]),
				float(det[2]) + float(det[4]),
				float(det[3]) + float(det[5])]),
				"conf": float(det[6]),
				"difficult": False
			})

	return detections


def get_rect_iou(a, b):
	"""Return iou for a single a pair of boxes"""
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


def get_frame_iou(gt_rects, det_rects):
	"""Return iou for a frame"""
	list_iou = []

	for gt in gt_rects:
		max_iou = 0
		for obj in det_rects:
			det = obj['bbox']
			iou = get_rect_iou(det, gt)
			if iou > max_iou:
				max_iou = iou

		if max_iou != 0:
			list_iou.append(max_iou)

	return np.mean(list_iou)

def display_frame(frame, annotations, detections):

	for r in annotations:
		frame = cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)

	for r in detections:
		bbox = r['bbox']
		frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

	imS = cv2.resize(frame, (960, 540))
	cv2.imshow('Frame', imS)
	cv2.waitKey(0)  # waits until a key is pressed
	cv2.destroyAllWindows()
