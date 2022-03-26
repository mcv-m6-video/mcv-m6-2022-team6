import argparse
import numpy as np

import Week1.utils_week1 as uw1
from TrackingMov import TrackingMov
import cv2

annotations_path = '../data/ai_challenge_s03_c010-full_annotation.xml'
TOTAL_FRAMES = 2141


def draw_bbox(img, bbox, id=1, color=(0, 0, 255)):
	bbox = np.ceil(bbox).astype(np.int32);
	img = cv2.putText(img, "Id %d" % id, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
	return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Car tracking")
	parser.add_argument('--iou_th', default=0.5)
	parser.add_argument('--tracker', default='mov')
	args = parser.parse_args()

	annotations = uw1.read_annotations(annotations_path, False);

	if args.tracker == "mov":
		tracker = TrackingMov(args.iou_th);
	else:
		tracker = TrackingMov(args.iou_th);

	for i in range(TOTAL_FRAMES):
		frameId = i + 1;
		print("Frame %04d of %04d" % (frameId, TOTAL_FRAMES));
		img = cv2.imread("../data/images/%04d.jpeg" % frameId);
		bboxes = annotations[i]

		tracks = tracker.generate_track(i, bboxes)
		if len(tracks) > 0:
			for track in tracks:
				img = draw_bbox(img, track['bbox'], track["id"])

		# Show
		cv2.imshow('Video', img)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
