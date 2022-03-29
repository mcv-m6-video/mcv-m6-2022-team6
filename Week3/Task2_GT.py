import argparse
import numpy as np

import utils_week3 as uw3
from Tracking import TrackingIOU, TrackingKalman, TrackingKalmanSort
import cv2

annotations_path = '../data/ai_challenge_s03_c010-full_annotation.xml'
TOTAL_FRAMES = 2141
SHOW_VIDEO = False


def draw_bbox(img, bbox, id=1, color=(0, 0, 255)):
	bbox = np.ceil(bbox).astype(np.int32);
	img = cv2.putText(img, "Id %d" % id, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
	return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Car tracking")
	parser.add_argument('--iou_th', default=0.4)
	parser.add_argument('--tracker', default='iou')
	parser.add_argument('--input', default='gt')
	args = parser.parse_args()

	if args.input == "gt":
		gt_ids, annotations = uw3.read_annotations(annotations_path, False);
	else:
		gt_ids, annotations = uw3.read_annotations(annotations_path, True);

	if args.tracker == "iou":
		tracker = TrackingIOU(gt_ids, annotations, args.iou_th);
	elif args.tracker == "kalman":
		tracker = TrackingKalman(gt_ids, annotations, args.iou_th);
	elif args.tracker == "kalmansort":
		tracker = TrackingKalmanSort(gt_ids, annotations, args.iou_th)

	if args.input == "gt":
		detections = annotations
	elif args.input == "retinanet101":
		detections = uw3.read_detections('../data/detections/retinanet_R_50_FPN_3x/detections.txt');

	for i in range(228, TOTAL_FRAMES):
		frameId = i + 1;
		print("Frame %04d of %04d" % (frameId, TOTAL_FRAMES));
		img = cv2.imread("../data/images/%04d.jpeg" % frameId);
		bboxes = detections[i]

		tracks = tracker.generate_track(i, bboxes)
		if len(tracks) > 0:
			for track in tracks:
				img = draw_bbox(img, track['bbox'], track["id"], track['color'])

		# Show
		if SHOW_VIDEO:
			cv2.imshow('Video', cv2.resize(img, (1920//2, 1080//2)))
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

	results = tracker.get_IDF1()
	print(results);