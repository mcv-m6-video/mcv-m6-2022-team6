import argparse
import numpy as np
import sys
from Week3 import utils_week3 as uw3
import imageio
import cv2

from TrackingOF import TrackingOF
from Week3.Tracking import TrackingIOU

annotations_path = '../data/ai_challenge_s03_c010-full_annotation.xml'
TOTAL_FRAMES = 2141


def draw_bbox(img, bbox, id=1, color=(0, 0, 255)):
	bbox = np.ceil(bbox).astype(np.int32);
	img = cv2.putText(img, "Id %d" % id, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
	return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Car tracking")
	parser.add_argument('--iou_th', default=0.3)
	parser.add_argument('--input', default='faster101')
	parser.add_argument('--preview', default=True)
	args = parser.parse_args()

	if args.input == "gt":
		gt_ids, annotations = uw3.read_annotations(annotations_path, False);
	else:
		gt_ids, annotations = uw3.read_annotations(annotations_path, True);

	tracker = TrackingOF(gt_ids, annotations, args.iou_th)
	tracker2 = TrackingIOU(gt_ids, annotations, args.iou_th)

	if args.input == "gt":
		detections = annotations
	elif args.input == "retinanet50":
		detections = uw3.read_detections('../data/detections/retinanet_R_50_FPN_3x/detections.txt');
	elif args.input == "faster50":
		detections = uw3.read_detections('../data/detections/faster_rcnn_R_50_FPN_3x/detections.txt');
	elif args.input == "retinanet101":
		detections = uw3.read_detections('../data/detections/retinanet_R_101_FPN_3x/detections.txt');
	elif args.input == "faster101":
		detections = uw3.read_detections('../data/detections/faster_rcnn_X_101_32x8d_FPN_3x/detections.txt');

	# writer = imageio.get_writer("results/task22_overlap_cars.gif", mode="I")
	for i in range(0, TOTAL_FRAMES):
		frameId = i + 1;
		if i % 100 == 0:
			print("Frame %04d of %04d" % (frameId, TOTAL_FRAMES))

		img = cv2.imread("../data/images/%04d.jpeg" % frameId)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
		bboxes = detections[i]

		
		#if frameId > 168:
		#	print('vdfvdfvfd')
		tracks, mask = tracker.generate_track(i, bboxes, img_gray)
		tracks = tracker2.generate_track(i, bboxes)
		if args.preview:
			if len(tracks) > 0:
				for track in tracks:
					img = draw_bbox(img, track['bbox'], track["id"], track['color'])

			img = cv2.add(img, mask.astype(np.uint8))

			# Show
			# writer.append_data(cv2.cvtColor(cv2.resize(img, (1920//2, 1080//2)), cv2.COLOR_BGR2RGB))
			cv2.imshow('Video', cv2.resize(img, (1920 // 2, 1080 // 2)))
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
			#cv2.imwrite("../data/results_optical/%04d.jpeg" % frameId,img)


	# writer.close()
	results = tracker.get_IDF1()
	print(results)
