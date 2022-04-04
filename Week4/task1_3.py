import argparse
import numpy as np

from Week3 import utils_week3 as uw3
from Week3.Tracking import TrackingIOU, TrackingKalman, TrackingKalmanSort, TrackingIOUDirection
import cv2
from lib import pyflow
import imageio
import cv2

annotations_path = '../data/ai_challenge_s03_c010-full_annotation.xml'
TOTAL_FRAMES = 2141

# Flow Options:
alpha = 0.1
ratio = 0.75
minWidth = 20
nOuterFPIterations = 2
nInnerFPIterations = 1
nSORIterations = 2
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


def draw_bbox(img, bbox, id=1, color=(0, 0, 255)):
	bbox = np.ceil(bbox).astype(np.int32);
	img = cv2.putText(img, "Id %d" % id, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
	return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

def to_flow_view(img, flow):
	hsv = np.zeros(img.shape, dtype=np.uint8)
	hsv[:, :, 0] = 255
	hsv[:, :, 1] = 255
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Car tracking")
	parser.add_argument('--iou_th', default=0.3)
	parser.add_argument('--tracker', default='kalmansort')
	parser.add_argument('--input', default='faster101')
	parser.add_argument('--preview', default=True)
	args = parser.parse_args()

	if args.input == "gt":
		gt_ids, annotations = uw3.read_annotations(annotations_path, False);
	else:
		gt_ids, annotations = uw3.read_annotations(annotations_path, True);

	if args.tracker == "iou":
		tracker = TrackingIOU(gt_ids, annotations, args.iou_th);
	elif args.tracker == "iou_direction":
		tracker = TrackingIOUDirection(gt_ids, annotations, args.iou_th);
	elif args.tracker == "kalman":
		tracker = TrackingKalman(gt_ids, annotations, args.iou_th);
	elif args.tracker == "kalmansort":
		tracker = TrackingKalmanSort(gt_ids, annotations, args.iou_th)

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
	prev_frame = cv2.imread("../data/images/%04d.jpeg" % 1, cv2.IMREAD_GRAYSCALE)
	for i in range(1, TOTAL_FRAMES):
		frameId = i + 1;
		if i % 100 == 0:
			print("Frame %04d of %04d" % (frameId, TOTAL_FRAMES))

		img = cv2.imread("../data/images/%04d.jpeg" % frameId)
		bboxes = detections[i]

		if not prev_frame is None:
			u, v, im2W = pyflow.coarse2fine_flow(prev_frame/255.,
												img/255.,
												 alpha, ratio, minWidth, nOuterFPIterations,
												 nInnerFPIterations,
												 nSORIterations, colType)
			flow = np.concatenate((u[..., None], v[..., None]), axis=2)
			img_flow = to_flow_view(img, flow)

		tracks = tracker.generate_track(i, bboxes)
		prev_frame = cv2.cvtColor(img.copy(), cv2.COLOR_BGRA2GRAY);
		if args.preview:
			if len(tracks) > 0:
				for track in tracks:
					img = draw_bbox(img, track['bbox'], track["id"], track['color'])

			# Show
			# writer.append_data(cv2.cvtColor(cv2.resize(img, (1920//2, 1080//2)), cv2.COLOR_BGR2RGB))
			cv2.imshow('Video', cv2.resize(img_flow, (1920 // 2, 1080 // 2)))
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

	# writer.close()
	results = tracker.get_IDF1()
	print(results);
