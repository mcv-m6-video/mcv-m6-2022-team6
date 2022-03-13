import xmltodict

def read_annotations(path):
	with open(path) as f:
		tracks = xmltodict.parse(f.read())['annotations']['track']

	annotations = {}
	for track in tracks:
		id = track['@id']
		label = track['@label']

		if label != 'car':
			continue

		for box in track['box']:
			frame = int(box['@frame'])
			if frame not in annotations:
				annotations[frame] = []

			annotations[frame].append([
				float(box['@xtl']),
				float(box['@ytl']),
				float(box['@xbr']),
				float(box['@ybr']),
			])
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

			detections[frame].append([
				float(det[2]),
				float(det[3]),
				float(det[2]) + float(det[4]),
				float(det[3]) + float(det[5]),
				float(det[6])
			])

	return detections
