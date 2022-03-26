import Week1.utils_week1 as uw1


class TrackingMov(object):
	_tracks = {}
	_trackingId = 1;

	def __init__(self, iou_th=0.5):
		self._iou_th = iou_th;

	def generate_track(self, frame, bboxes):

		self._tracks[frame] = [];
		for bbox in bboxes:
			best_track = None
			best_iou = 0
			for track in self._tracks[frame - 1]:
				iou = uw1.get_rect_iou(bbox, track['bbox'])
				if iou >= self._iou_th and best_iou < iou:
					best_track = track

			if not best_track:
				assignedId = self._trackingId
				self._trackingId = self._trackingId + 1
				track = {"id": assignedId, "bbox": bbox};
			else:
				track = best_track
				track['bbox'] = bbox;

			self._tracks[frame].append(track)

		return self._tracks[frame]
