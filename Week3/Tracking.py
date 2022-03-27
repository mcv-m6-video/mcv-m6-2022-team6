import abc
from filterpy.kalman import KalmanFilter

import Week1.utils_week1 as uw1
import numpy as np


class TrackingBase(object):
	_tracks = {}
	_trackingId = 1;
	_colours = np.random.rand(200, 3) * 255
	_last_frame = []

	def __init__(self, iou_th=0.5):
		self._iou_th = iou_th;

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


class TrackingIOU(TrackingBase):

	def __init__(self, iou_th=0.5):
		super().__init__(iou_th)

	def generate_track(self, frame, bboxes):

		new_frame = []
		for bbox in bboxes:
			best_track = self.find_best_iou(bbox, self._last_frame)

			if not best_track:
				assignedId = self.generate_id();
				self._tracks[assignedId] = [];
				newTrack = {"id": assignedId, "frame": frame, "bbox": bbox,
							 "color": self._colours[assignedId - 1]}
				self._tracks[assignedId].append(newTrack)
				new_frame.append(newTrack);
			else:
				best_track['bbox'] = bbox;
				best_track['frame'] = frame;
				self._tracks[best_track['id']].append(best_track)
				new_frame.append(best_track);

		self._last_frame = new_frame;

		return new_frame


class TrackingKalman(TrackingBase):

	def __init__(self, iou_th=0.5):
		super().__init__(iou_th)
		self.kalman_tracking = {};

	def generate_track(self, frame, bboxes):

		if not (frame - 1) in self._tracks:
			self._tracks[frame - 1] = []

		self._tracks[frame] = [];
		tracks_found = []
		for bbox in bboxes:
			track_selected = self.find_best_iou(bbox, self._tracks[frame - 1])

			if track_selected:
				trackId = track_selected['id'];
				tracks_found.append(trackId)
				track_selected['bbox'] = bbox
				self.kalman_tracking[trackId].update_state(frame, bbox);
			else:
				trackId = self.generate_id();
				self.kalman_tracking[trackId] = TrackingKalmanItem(bbox);
				track_selected = {"id": trackId, "bbox": bbox,
								  "color": self._colours[trackId]};

			self._tracks[frame].append(track_selected);

		for key, value in self.kalman_tracking.items():
			if not key in tracks_found:
				bbox = value.update(frame)[1];
				print(bbox)
				if bbox[0] > 0 and bbox[1] > 0 and bbox[0] + bbox[2] > 100 \
						and bbox[1] + bbox[3] > 100:
					track_selected = {"id": key, "bbox": bbox,
									  "color": self._colours[key]};
					self._tracks[frame].append(track_selected);

		return self._tracks[frame]


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