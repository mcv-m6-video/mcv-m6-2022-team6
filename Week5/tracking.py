import abc
import copy

from sort import Sort
import numpy as np
import motmetrics as mm
from utils import get_rect_iou


class TrackingBase(object):
	_tracks = {}
	_trackingId = 1;
	_colours = np.random.rand(2000, 3) * 255
	_last_frame = []

	# Create an accumulator that will be updated during each frame
	_acc = mm.MOTAccumulator(auto_id=True)
	_gt_ids = []
	_gt_bbox = [];

	def __init__(self, gt_ids, gt_bbox, iou_th=0.5):
		self._iou_th = float(iou_th);
		self._gt_ids = gt_ids;
		self._gt_bbox = gt_bbox;

	@abc.abstractmethod
	def generate_track(self, frame, bboxes):
		raise NotImplementedError("Not implemented this method")

	def unit_vector(self, vector):
		""" Returns the unit vector of the vector.  """
		return vector / np.linalg.norm(vector)

	def angle_between(self, v1, v2):
		v1_u = self.unit_vector(v1)
		v2_u = self.unit_vector(v2)
		return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

	def find_best_iou(self, bbox, tracks, ids_selected, direction=False):
		best_track = None
		best_iou = 0
		center = (bbox[0:2] + bbox[2:4])//2
		for track in tracks:
			if track['id'] in ids_selected:
				continue

			iou = get_rect_iou(bbox, track['bbox'])
			if iou >= self._iou_th and best_iou < iou:
				if direction and track['angle'] is not None \
						and np.linalg.norm(self.angle_between(center, track['center']) - track['angle']) > 0.5:
					continue

				best_track = copy.deepcopy(track)


		return best_track

	def generate_id(self):
		assignedId = self._trackingId
		self._trackingId += 1
		return assignedId

	def update_metrics(self, frame, tracks_prev):

		detected_ids = []
		gt_ids = self._gt_ids[frame];
		gt_bboxes = np.array(self._gt_bbox[frame]).astype(np.int32);
		distances = []
		for track in tracks_prev:
			distances_track = [];
			detected_ids.append(track['id']);
			for bbox in gt_bboxes:
				distances_track.append(get_rect_iou(bbox[0:4], track['bbox']))

			distances.append(distances_track)

		self._acc.update(gt_ids, detected_ids, distances)

	def get_IDF1(self):
		mh = mm.metrics.create()
		return mh.compute(self._acc, metrics=['num_frames', 'idf1'], name='acc')


class TrackingKalmanSort(TrackingBase):

	def __init__(self, gt_ids, gt_bbox, iou_th=0.5):
		super().__init__(gt_ids, gt_bbox, float(iou_th))
		self.sort = Sort(iou_threshold=float(iou_th), max_age=5)

	def generate_track(self, frame, bboxes):

		if len(bboxes) == 0:
			bboxes = np.empty((0, 5))

		predicted = self.sort.update(bboxes)

		new_frame = []
		for i in range(len(predicted)):
			tracker = self.sort.trackers[i]
			bbox = self.sort.trackers[i].get_state()[0]
			newTrack = {"id": tracker.id, "frame": frame, "bbox": bbox.astype(np.int32),
						"color": self._colours[tracker.id]}
			new_frame.append(newTrack)

		self.update_metrics(frame, self._last_frame)
		self._last_frame = new_frame

		return new_frame
