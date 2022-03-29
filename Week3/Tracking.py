import abc

import Week1.utils_week1 as uw1
from sort import *
import motmetrics as mm


class TrackingBase(object):
	_tracks = {}
	_trackingId = 1;
	_colours = np.random.rand(1000, 3) * 255
	_last_frame = []

	# Create an accumulator that will be updated during each frame
	_acc = mm.MOTAccumulator(auto_id=True)
	_gt_ids = []
	_gt_bbox = [];

	def __init__(self, gt_ids, gt_bbox, iou_th=0.5):
		self._iou_th = iou_th;
		self._gt_ids = gt_ids;
		self._gt_bbox = gt_bbox;

	@abc.abstractmethod
	def generate_track(self, frame, bboxes):
		raise NotImplementedError("Not implemented this method")

	def find_best_iou(self, bbox, tracks):
		best_track = None
		best_iou = 0
		for track in tracks:
			iou = uw1.get_rect_iou(bbox, track['bbox'])
			if iou >= self._iou_th and best_iou < iou:
				best_track = track

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
				distances_track.append(uw1.get_rect_iou(bbox[0:4], track['bbox']))

			distances.append(distances_track)

		self._acc.update(gt_ids, detected_ids, distances)

	def get_IDF1(self):
		mh = mm.metrics.create()
		return mh.compute(self._acc, metrics=['num_frames', 'idf1'], name='acc')


class TrackingIOU(TrackingBase):

	def __init__(self, gt_ids, gt_bbox, iou_th=0.5):
		super().__init__(gt_ids, gt_bbox, iou_th)

	def generate_track(self, frame, bboxes):

		new_frame = []
		for bbox in bboxes:
			bbox_norm = bbox[0:4].astype(np.int32)
			track_selected = self.find_best_iou(bbox_norm, self._last_frame)

			if track_selected:
				track_selected['bbox'] = bbox_norm;
				track_selected['frame'] = frame;
				self._tracks[track_selected['id']].append(track_selected)
				new_frame.append(track_selected);
			else:
				assignedId = self.generate_id();
				self._tracks[assignedId] = [];
				newTrack = {"id": assignedId, "frame": frame, "bbox": bbox_norm,
							"color": self._colours[assignedId - 1]}
				self._tracks[assignedId].append(newTrack)
				new_frame.append(newTrack);

		self.update_metrics(frame, self._last_frame)
		self._last_frame = new_frame;

		return new_frame


class TrackingKalman(TrackingBase):

	def __init__(self, gt_ids, gt_bbox, iou_th=0.5):
		super().__init__(gt_ids, gt_bbox, iou_th)
		self.kalman_tracking = {};

	def generate_track(self, frame, bboxes):

		new_frame = []
		tracks_found = []
		for bbox in bboxes:
			bbox_norm = bbox[0:4].astype(np.int32);
			track_selected = self.find_best_iou(bbox_norm, self._last_frame)
			if track_selected:
				trackId = track_selected['id'];
				tracks_found.append(trackId)
				track_selected['bbox'] = bbox_norm;
				track_selected['frame'] = frame;
				self._tracks[trackId].append(track_selected)
				new_frame.append(track_selected);
				self.kalman_tracking[trackId].update_state(frame, bbox);
			else:
				trackId = self.generate_id();
				self.kalman_tracking[trackId] = TrackingKalmanItem(bbox);
				self._tracks[trackId] = [];
				newTrack = {"id": trackId, "frame": frame, "bbox": bbox_norm,
							"color": self._colours[trackId - 1]}
				self._tracks[trackId].append(newTrack)
				new_frame.append(newTrack);

		for trackId, value in self.kalman_tracking.items():
			if not trackId in tracks_found:
				bbox = value.update(frame)[1].astype(np.int32);
				lastbbox = self._tracks[trackId][-1]['bbox'];
				if bbox[0] > 0 and bbox[1] > 0 and bbox[0] + bbox[2] > 100 \
						and bbox[1] + bbox[3] > 100 and uw1.get_rect_iou(bbox, lastbbox) < 0.9:
					newTrack = {"id": trackId, "frame": frame, "bbox": bbox,
								"color": self._colours[trackId - 1]}
					new_frame.append(newTrack);
					self._tracks[trackId].append(newTrack)

		self.update_metrics(frame, self._last_frame)
		self._last_frame = new_frame;

		return new_frame


class TrackingKalmanItem(object):

	def __init__(self, bbox):
		# define constant velocity model
		self.kf = KalmanFilter(dim_x=7, dim_z=4)
		self.kf.F = np.array(
			[[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
			 [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
		self.kf.H = np.array(
			[[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

		self.kf.R[2:, 2:] *= 10.
		self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
		self.kf.P *= 10.
		self.kf.Q[-1, -1] *= 0.01
		self.kf.Q[4:, 4:] *= 0.01

		self.kf.x[:4] = self.convert_bbox_to_z(bbox)
		self.bbox = bbox

		self.history = []
		self.age = 0

	def update(self, frame):
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		if ((self.kf.x[6] + self.kf.x[2]) <= 0):
			self.kf.x[6] *= 0.0
		self.kf.predict()
		self.age += 1
		predicted = self.convert_x_to_bbox(self.kf.x)
		# print(f"Predicted: {predicted}")
		self.history.append(predicted)
		return True, predicted[0]

	def update_state(self, frame, bbox):
		xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
		self.history = []
		new_bbox = self.convert_bbox_to_z(xywh)
		# print(f"NEW: {new_bbox}")
		self.kf.update(new_bbox)

	def convert_bbox_to_z(self, bbox):
		"""
		Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
		  [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
		  the aspect ratio
		"""
		w = bbox[2]  # - bbox[0]
		h = bbox[3]  # - bbox[1]
		x = bbox[0] + w / 2.
		y = bbox[1] + h / 2.
		s = w * h  # scale is just area
		r = w / float(h)
		return np.array([x, y, s, r]).reshape((4, 1))

	def convert_x_to_bbox(self, x, score=None):
		"""
		Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
		[x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
		"""
		w = np.sqrt(x[2] * x[3])
		h = x[2] / w
		if (score == None):
			xyxy = [x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]
		else:
			xyxy = [x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]

		return np.array([xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]).reshape((1, len(xyxy)))


class TrackingKalmanSort(TrackingBase):

	def __init__(self, gt_ids, gt_bbox, iou_th=0.5):
		super().__init__(gt_ids, gt_bbox, iou_th)
		self.sort = Sort(iou_threshold=iou_th)

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
