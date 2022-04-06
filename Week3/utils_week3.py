import numpy as np
import cv2
import pandas
import xmltodict

colors = {
    'r': (0, 0, 255),
    'g': (0, 255, 0),
    'b': (255, 0, 0),
    'w': (255,255,255)
}
color_ids = {}

def draw_boxes(image, boxes, tracker=None, color='g', linewidth=2, det=False, boxIds=False, old=False):
    rgb = colors[color]
    for box in boxes:
        # print(box.id)
        if boxIds:
            if box['id'] in list(color_ids.keys()):
                pass
            else:
                color_ids[box['id']]=np.random.uniform(0,256,size=3)
            if old:
                cv2.putText(image, str(box['id']), (int(box['bbox'][0]), int(box['bbox'][1]) + 120), cv2.FONT_ITALIC, 0.6, color_ids[box['id']], linewidth)
            else:
                cv2.putText(image, str(box['id']), (int(box['bbox'][0]), int(box['bbox'][1]) + 20), cv2.FONT_ITALIC, 0.6, color_ids[box['id']], linewidth)

            if tracker is not None:
                if box['id'] in tracker:
                    if len(tracker[box['id']])>2:
                        image =cv2.polylines(image,[np.array(tracker[box['id']])],False,color_ids[box['id']],linewidth)

            # if len(kalman_predictions[box.id])>2:
            #     image =cv2.polylines(image,[np.array(kalman_predictions[box.id])],False,color_ids[box.id],linewidth)


            image = cv2.rectangle(image, (int(box['bbox'][0]), int(box['bbox'][1])), (int(box['bbox'][2]), int(box['bbox'][3])), color_ids[box['id']], linewidth)
        else:
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), rgb, linewidth)

        if det:
            cv2.putText(image, str(box['confidence']), (int(box['bbox'][0]), int(box['bbox'][1]) - 5), cv2.FONT_ITALIC, 0.6, rgb, linewidth)
       
    return image
	

def draw_boxes1(image, boxes, tracker=None, color='g', linewidth=2, det=False, boxIds=False, old=False):
    rgb = colors[color]
    for box in boxes:
        # print(box.id)
        if boxIds:
            if box['id'] in list(color_ids.keys()):
                pass
            else:
                color_ids[box['id']]=np.random.uniform(0,256,size=3)
            if old:
                cv2.putText(image, str(box['id']), (int(box['bbox'][0]), int(box['bbox'][1]) + 120), cv2.FONT_ITALIC, 0.6, color_ids[box['id']], linewidth)
            else:
                cv2.putText(image, str(box['id']), (int(box['bbox'][0]), int(box['bbox'][1]) + 20), cv2.FONT_ITALIC, 0.6, color_ids[box['id']], linewidth)

            if tracker is not None:
                if box['id'] in tracker:
                    if len(tracker[box['id']])>2:
                        image =cv2.polylines(image,[np.array(tracker[box['id']])],False,color_ids[box['id']],linewidth)

            # if len(kalman_predictions[box.id])>2:
            #     image =cv2.polylines(image,[np.array(kalman_predictions[box.id])],False,color_ids[box.id],linewidth)


            image = cv2.rectangle(image, (int(box['bbox'][0]), int(box['bbox'][1])), (int(box['bbox'][2]), int(box['bbox'][3])), color_ids[box['id']], linewidth)
        else:
            image = cv2.rectangle(image, (int(box['bbox'][0]), int(box['bbox'][1])), (int(box['bbox'][2]), int(box['bbox'][3])), rgb, linewidth)

        if det:
            cv2.putText(image, str(box['confidence']), (int(box['bbox'][0]), int(box['bbox'][1]) - 5), cv2.FONT_ITALIC, 0.6, rgb, linewidth)
       
    return image


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
		annotations[i] = []  # creates an empy list equal to the number of frames (2141)
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
			annotations[frame].append(
				np.array([float(box['@xtl']), float(box['@ytl']), float(box['@xbr']), float(box['@ybr']), 1]))

	return gt_ids, annotations

def read_detections(file_name, iou_th=0.5):
	output = {}
	detections = pandas.read_csv(file_name, delimiter=",", header=None).to_numpy()
	for detection in detections:
		frameId = int(detection[0]) - 1
		if not frameId in output:
			output[frameId] = []

		if detection[6] >= iou_th:
			bbox = np.hstack((detection[2:4], detection[2:4] + detection[4:6], detection[6]))
			output[frameId].append(bbox)

	return output

def read_detections1(file_name, nframes, iou_th=0.5):
	annotations = {}
	gt_ids ={}
	for i in range(nframes):
		annotations[i] = []  # creates an empy list equal to the number of frames
		gt_ids[i] = []
	
	detections = pandas.read_csv(file_name, delimiter=",", header=None).to_numpy()
	
	for detection in detections:
		frameId = int(detection[0]) - 1
		if not frameId in annotations:
			annotations[frameId] = []
			gt_ids[frameId] = []

		if detection[6] >= iou_th:
			bbox = np.hstack((detection[2:4], detection[2:4] + detection[4:6], detection[6]))
			id = int(detection[1])
			annotations[frameId].append(bbox)
			gt_ids[frameId].append(id)
	return gt_ids, annotations

def read_detections2(file_name, iou_th=0.5):
	output = {}
	detections = pandas.read_csv(file_name, delimiter=",", header=None).to_numpy()
	for detection in detections:
		frameId = int(detection[0]) - 1
		if not frameId in output:
			output[frameId] = []

		if detection[6] >= iou_th:
			bbox = np.hstack((detection[2:4], detection[2:4] + detection[4:6], detection[6]))
			id = int(detection[1])
			output[frameId].append({'bbox': bbox, 'id': id})

	return output
