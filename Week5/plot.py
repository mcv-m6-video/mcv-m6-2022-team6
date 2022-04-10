import cv2
import numpy as np

PREVIEW_SIZE = (1920 // 3, 1080 // 3)

def draw_bbox(img, bbox, id=1, color=(0, 0, 255)):
	bbox = np.ceil(bbox).astype(np.int32);
	img = cv2.putText(img, "Id %d" % id, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
	return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

def draw_detections(img, detections):
	for det in detections:
		img = draw_bbox(img, det[0:4])

	return img


def plot_camera(name, frame, detections):
	img = draw_detections(frame, detections)
	img = cv2.resize(img, PREVIEW_SIZE)
	cv2.imshow(name, img)
