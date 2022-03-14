import xmltodict
import numpy as np

def read_annotations(path):
	with open(path) as f:
		data = xmltodict.parse(f.read())
		tracks = data['annotations']['track']
		nframes = int(data['annotations']['meta']['task']['size'])

	annotations = {}
	for i in range(nframes + 1):
		annotations[i] = [];

	for track in tracks:
		id = track['@id']
		label = track['@label']

		if label != 'car':
			continue

		for box in track['box']:
			frame = int(box['@frame'])
			annotations[frame].append(np.array([
				float(box['@xtl']),
				float(box['@ytl']),
				float(box['@xbr']),
				float(box['@ybr']),
			]))
	return annotations

def read_detections(path, confidenceThr=0.5):
	"""
	Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
	"""
	with open(path) as f:
		lines = f.readlines()

	detections = {}
	for line in lines:
		det = line.split(sep=',')
		if float(det[6]) > confidenceThr:

			frame = int(det[0])
			if frame not in detections:
				detections[frame] = []

			detections[frame].append({
				"bbox": np.array([float(det[2]),
				float(det[3]),
				float(det[2]) + float(det[4]),
				float(det[3]) + float(det[5])]),
				"conf": float(det[6]),
				"difficult": False
			})

	return detections
