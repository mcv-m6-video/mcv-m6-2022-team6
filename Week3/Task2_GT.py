import argparse
import numpy as np

import Week1.utils_week1 as uw1
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
	args = parser.parse_args()

	annotations = uw1.read_annotations(annotations_path, False);
	tracks = {}
	trackingId = 1;

	for i in range(TOTAL_FRAMES):
		frameId = i + 1;
		print("Frame %04d of %04d" % (frameId, TOTAL_FRAMES));
		img = cv2.imread("../data/images/%04d.jpeg" % frameId);

		bboxes = annotations[i]
		tracks[i] = [];
		if len(bboxes) > 0:
			for bbox in bboxes:
				best_track = None
				best_iou = 0
				for track in tracks[i - 1]:
					iou = uw1.get_rect_iou(bbox, track['bbox'])
					if iou >= args.iou_th and best_iou < iou:
						best_track = track

				if not best_track:
					assignedId = trackingId
					trackingId = trackingId + 1
					track = {"id": assignedId, "bbox": bbox};
				else:
					track = best_track
					track['bbox'] = bbox;

				tracks[i].append(track)

				img = draw_bbox(img, bbox, track["id"])

		# Show
		cv2.imshow('Video', img)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
