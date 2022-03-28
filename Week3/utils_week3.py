import numpy as np
import xmltodict


def read_annotations(path, use_parked=False):

	idTracking = 1
	idsAssigned = {}
	with open(path) as f:
		data = xmltodict.parse(f.read())
		tracks = data['annotations']['track']
		nframes = int(data['annotations']['meta']['task']['size'])

	annotations = {}
	gt_ids = {}
	for i in range(nframes):
		annotations[i] = []#creates an empy list equal to the number of frames (2141)
		gt_ids[i] = []

	for track in tracks:
		id = track['@id']
		label = track['@label']

		if label != 'car':
			continue

		for box in track['box']:
			is_parked = box['attribute']['#text'] == 'true'
			if not use_parked and is_parked:
				continue
			
			frame = int(box['@frame'])
			if not id in idsAssigned:
				idsAssigned[id] = idTracking
				idTracking += 1

			effectiveId = idsAssigned[id]
			gt_ids[frame].append(effectiveId)
			annotations[frame].append(np.array([float(box['@xtl']), float(box['@ytl']), float(box['@xbr']), float(box['@ybr'])]))

	return gt_ids, annotations