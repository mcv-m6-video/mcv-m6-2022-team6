import xmltodict
import numpy as np
from torchvision.io import read_video
import cv2
import random
from matplotlib import pyplot as plt

def annotations_to_detections(annotations, noisy=False, noisy_p=0.5, dropout=False, dropout_p=0.5):

	detections = {}
	for frame, value in annotations.items():
		detections[frame] = []
		for bbox in value:
			r_noise = random.randint(0, 100) / 100
			r_dropout = random.randint(0, 100) / 100
			if dropout and r_dropout <= dropout_p:
				continue

			bboxcpy = np.copy(bbox)
			print(r_noise)
			if noisy and r_noise <= noisy_p:
				rx = random.randint(500, 2000) / 100
				ry = random.randint(500, 2000) / 100
				bboxcpy[0] = bbox[0] + rx;
				bboxcpy[1] = bbox[1] + ry;
				bboxcpy[2] = bbox[2] + rx;
				bboxcpy[3] = bbox[3] + ry;

			detections[frame].append({
				"bbox": bboxcpy,
				"conf": 1,
				"difficult": False
			})

	return detections


def read_frame(video, frame=0):
	frame = read_video(video, frame, frame)[0]
	img = frame.reshape(frame.shape[1:]).numpy();
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

def read_frame_bw(video, frame =0):
	frame = read_video(video, frame, frame)[0]
	img = frame.reshape(frame.shape[1:]).numpy();
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

def read_annotations(path):
	with open(path) as f:
		data = xmltodict.parse(f.read())
		tracks = data['annotations']['track']
		nframes = int(data['annotations']['meta']['task']['size'])

	annotations = {}
	for i in range(nframes):
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
			if frame - 1 not in detections:
				detections[frame - 1] = []

			detections[frame - 1].append({
				"bbox": np.array([float(det[2]),
				float(det[3]),
				float(det[2]) + float(det[4]),
				float(det[3]) + float(det[5])]),
				"conf": float(det[6]),
				"difficult": False
			})

	return detections


def get_rect_iou(a, b):
	"""Return iou for a single a pair of boxes"""
	x11, y11, x12, y12 = a
	x21, y21, x22, y22 = b

	xA = max(x11, x21)
	yA = max(y11, y21)
	xB = min(x12, x22)
	yB = min(y12, y22)

	# respective area of ​​the two boxes
	boxAArea = (x12 - x11) * (y12 - y11)
	boxBArea = (x22 - x21) * (y22 - y21)

	# overlap area
	interArea = max(xB - xA, 0) * max(yB - yA, 0)

	# IOU
	return interArea / (boxAArea + boxBArea - interArea)


def get_frame_iou(gt_rects, det_rects):
	"""Return iou for a frame"""
	list_iou = []

	for gt in gt_rects:
		max_iou = 0
		for obj in det_rects:
			det = obj['bbox']
			iou = get_rect_iou(det, gt)
			if iou > max_iou:
				max_iou = iou

		if max_iou != 0:
			list_iou.append(max_iou)

	return np.mean(list_iou)

def print_bboxes(frame, annotations, detections):

	for r in annotations:
		frame = cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)

	for r in detections:
		bbox = r['bbox']
		frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

	return frame;


def display_frame(frame):
	imS = cv2.resize(frame, (960, 540))
	cv2.imshow('Frame', imS)
	cv2.waitKey(0)  # waits until a key is pressed
	cv2.destroyAllWindows()

def lucasKanade(imgA, imgB):
	#u = (A^t A)^-1 A^T B
	flow = np.invert(np.transpose(imgA) * imgA)*np.transpose(imgA)*imgB
	flow = imgA-imgB
	return flow

def flow_read(flow_dir):
    # cv2 imread ---> BGR  need to converted in RGB format

    im = cv2.imread(flow_dir, cv2.IMREAD_UNCHANGED)
    im_kitti = np.flip(im, axis=2).astype(np.double)
    # As Lucas-Kanade follows the formula [u v]^T= (A^t A)^-1 A^T B 
	# We normalize x and y component that are divided in 2 differnt layers
    u_f = (im_kitti[:, :, 0] - 2. ** 15) / 64
    v_f = (im_kitti[:, :, 1] - 2. ** 15) / 64

    # All the points with module smaller than 1 wil be considered as static
    f_valid = im_kitti[:, :, 2]
    f_valid[f_valid > 1] = 1

    u_f[f_valid == 0] = 0
    v_f[f_valid == 0] = 0

    #flow = np.dstack([u_f, v_f])

    flow_1 = np.dstack([u_f, v_f, f_valid])

    return flow_1

def msen(F_gt, F_test):
    SEN = []

    E_du = (F_gt[:, :, 0] - F_test[:, :, 0])

    E_dv = (F_gt[:, :, 1] - F_test[:, :, 1])

    E = np.sqrt(E_du ** 2 + E_dv ** 2)

    F_valid_gt = F_gt[:, :, 2]

    E[F_valid_gt == 0] = 0  # 0s in ocluded pixels

    SEN = np.append(SEN, E[F_valid_gt != 0])  # take in account the error of the non-ocluded pixels

    MSEN = np.mean(SEN)

    plt.figure(1)
    plt.hist(E[F_valid_gt == 1], bins=40, density=True)
    plt.title('Optical Flow error')
    plt.xlabel('MSEN')
    plt.ylabel('Number of pixels')
    plt.savefig('histogram.png')
    plt.show()

    plt.figure(2)
    plt.imshow(np.reshape(E, F_gt.shape[:-1]))
    plt.colorbar()
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.savefig('error_image.png')
    plt.show()

    return MSEN