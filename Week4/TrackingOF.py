import copy

import numpy as np
import sys
sys.path.append('/home/marcelo/Documents/Master_CV/M6/mcv-m6-2022-team6')
from Week3 import get_rect_iou
from Week3.Tracking import TrackingBase
import cv2

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
					  qualityLevel=0.3,
					  minDistance=7,
					  blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
				 maxLevel=2,
				 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class TrackingOF(TrackingBase):
	first_frame = True
	prev_frame = None
	mask = None
	p0 = None
	good_new = []
	good_old = []
	near_dis = 100

	def __init__(self, gt_ids, gt_bbox, iou_th=0.5):
		super().__init__(gt_ids, gt_bbox, iou_th)

	def opticalLukasKanadeInit(self, img_gray):
		self.prev_frame = img_gray
		self.p0 = cv2.goodFeaturesToTrack(self.prev_frame, mask=None, **feature_params)
		self.mask = np.zeros_like(img_gray)
		self.first_frame = False

	def opticalLukasKanade(self, img_gray):

		# calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, img_gray, self.p0, None, **lk_params)

		# Select good points
		if p1 is not None:
			self.good_new = p1[st == 1]
			self.good_old = self.p0[st == 1]

		# draw the tracks
		for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self._colours[i].tolist(), 2)

		self.p0 = self.good_new.reshape(-1, 1, 2)
		flow = (self.good_new - self.good_old);

		return np.ceil(flow), self.p0, self.mask

	def drawLukasKanade(self, img):

		for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
			a, b = new.ravel()
			img = cv2.circle(img, (int(a), int(b)), 5, self._colours[i].tolist(), -1)

		return img

	def iou_tracking(self, frame, bboxes):

		new_frame = []
		id_selected = []
		for bbox in bboxes:
			bbox_norm = bbox[0:4].astype(np.int32)
			track_selected = self.find_best_iou(bbox_norm, self._last_frame, id_selected)

			if track_selected:
				id_selected.append(track_selected['id'])
				track_selected['bbox'] = bbox_norm;
				track_selected['frame'] = frame;
				self._tracks[track_selected['id']].append(track_selected)
				new_frame.append(track_selected);
			else:
				assignedId = self.generate_id();
				self._tracks[assignedId] = [];
				newTrack = {"id": assignedId, "frame": frame, "bbox": bbox_norm,
							"color": self._colours[(assignedId - 1) % 2000],
							'center': (bbox_norm[0:2] + bbox_norm[2:4]) // 2}
				self._tracks[assignedId].append(newTrack)
				new_frame.append(newTrack);

		return new_frame

	def find_best_iou_of(self, bbox, tracks, ids_selected, flow):

		best_track = None
		best_tack_iou = 0

		for track in tracks:

			#if track['id'] in ids_selected:
			#		continue

			center = track['center']
			distance = np.sqrt(np.sum(np.square(center - self.p0), axis=2));
			near_points = distance < self.near_dis
			#if np.sum(track['bbox'] - bbox) == 0:
			#		best_track = track
			if np.sum(near_points) > 0:
				movement = flow[near_points.flatten(), :].mean(axis=0)

				# Generate temp bbox (move in flow direction)
				bbox_of = track['bbox'].copy()
				bbox_of[0:2] = bbox_of[0:2] + movement
				bbox_of[2:4] = bbox_of[2:4] + movement

				# Check if best
				iou = get_rect_iou(bbox, bbox_of);
				if 0.5 < iou and best_tack_iou < iou:
					best_track = track
					best_tack_iou = iou

		return best_track

	def generate_track(self, frame_id, bboxes, img_gray):

		new_frame = []
		id_selected = []

		if self.first_frame:
			self.opticalLukasKanadeInit(img_gray)
			new_frame = self.iou_tracking(frame_id, bboxes)
		else:
			flow, p0, mask = self.opticalLukasKanade(img_gray)

			for bbox in bboxes:
				bbox_norm = bbox.astype(np.int32)[0:4];
				best_track = self.find_best_iou_of(bbox_norm, self._last_frame, id_selected, flow)

				if best_track is None:
					assignedId = self.generate_id();
					self._tracks[assignedId] = [];
					newTrack = {"id": assignedId, "frame": frame_id, "bbox": bbox_norm,
								"color": self._colours[(assignedId - 1) % 2000],
								'center': (bbox_norm[0:2] + bbox_norm[2:4]) // 2}
					self._tracks[assignedId].append(newTrack)
					new_frame.append(newTrack)
				else:
					track_cpy = copy.deepcopy(best_track)
					id_selected.append(track_cpy['id'])
					track_cpy['bbox'] = bbox_norm
					track_cpy['frame'] = frame_id
					self._tracks[track_cpy['id']].append(track_cpy)
					new_frame.append(track_cpy)

		self.update_metrics(frame_id, self._last_frame)
		self._last_frame = new_frame

		return new_frame, self.mask
