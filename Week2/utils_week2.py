import xmltodict
import numpy as np
from torchvision.io import read_video
import cv2
import random
import matplotlib.pyplot as plt


def plot_gaussian_model(u, sigma):
	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1)
	imgplot = plt.imshow(u, cmap='gray')
	ax.set_title('Mean')
	ax = fig.add_subplot(1, 2, 2)
	imgplot = plt.imshow(sigma, cmap='gray')
	ax.set_title('Sigma')
	plt.show()

def annotations_to_detections(annotations, noisy=False, noisy_p=0.5, dropout=False, dropout_p=0.5,verbose=False):

	detections = {}
	for frame, value in annotations.items():
		detections[frame] = []
		for bbox in value:
			r_noise = random.randint(0, 100) / 100
			r_dropout = random.randint(0, 100) / 100
			if dropout and r_dropout <= dropout_p:
				continue

			bboxcpy = np.copy(bbox)
			if verbose:	
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

def get_frames(video, frame=0):
	return read_video(video)[0]


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
	len(lines)
	detections = {}
	dict_iter = 0
	for line in lines:
		det = line.split(sep=',')
		if float(det[6]) >= confidenceThr:

			frame = int(det[0])
			if frame - 1 not in detections:
				detections[frame - 1] = []
			detections[dict_iter] = []
			detections[dict_iter].append({
				"bbox": np.array([float(det[2]),
				float(det[3]),
				float(det[2]) + float(det[4]),
				float(det[3]) + float(det[5])]),
				"conf": float(det[6]),
				"difficult": False,
				"frame":int(det[0]) - 1
			})
			dict_iter += 1

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
	mean = np.mean(list_iou)
	if np.isnan(mean):
		return 0
	else:
		return mean

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

def plot_multiple_IoU(dict_of_detections,annotations):
	fig, ax = plt.subplots(figsize=(10, 5))
	detections = []
	score_for_each_frame = []
	for key in dict_of_detections:
		detections = read_detections(dict_of_detections[key])
		score_for_each_frame = []
		for frame_id in range(len(annotations)):
			score_for_each_frame.append(get_frame_iou(
            annotations[frame_id], detections[frame_id]))
		ax.plot(np.arange(0, len(annotations), 1), score_for_each_frame,label = key)
	ax.set(xlabel='frame', ylabel='IoU')
	ax.set_ylim(0, 1)
	ax.set_xlim(0,len(annotations))
	plt.legend()
	plt.savefig('comparison_plot.png')


def flow_read(flow_dir):
    # cv2 imread ---> BGR  need to converted in RGB format

    im_flow = cv2.imread(flow_dir, cv2.IMREAD_UNCHANGED)
    # As Lucas-Kanade follows the formula [u v]^T= (A^t A)^-1 A^T B 
	# We normalize x and y component that are divided in 2 differnt layers
    u_flow = (im_flow[:, :, 2] - 2. ** 15) / 64
    v_flow = (im_flow[:, :, 1] - 2. ** 15) / 64

    flow_valid = im_flow[:, :, 0]
    flow_valid[flow_valid > 1] = 1

    u_flow[flow_valid == 0] = 0
    v_flow[flow_valid == 0] = 0

    #flow = np.dstack([u_f, v_f])

    flow_map = np.dstack([u_flow, v_flow, flow_valid])

    plt.figure(1)
    plt.imshow(flow_map)
    plt.colorbar()
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.savefig('inputImage.png')
    plt.show()
    return flow_map

def msen(Frame_gt, Frame_test):
    '''
	Frame_gt : frame grountruth
	Frame_test: frame test
    '''
    SEN = []

    Module = np.sqrt(np.add(np.square(np.subtract(Frame_gt[:,:,0],Frame_test[:,:,0])),np.square(np.subtract(Frame_gt[:,:,1],Frame_test[:,:,1]))))

    F_valid_gt = Frame_gt[:, :, 2]

    Module[F_valid_gt == 0] = 0  # 0s in ocluded pixels

    SEN = np.append(SEN, Module[F_valid_gt != 0])  # take in account the error of the non-ocluded pixels

    MSEN = np.mean(SEN)

    plt.figure(1)
    plt.hist(Module[F_valid_gt == 1], bins=40, density=True)
    plt.title('Optical Flow error')
    plt.xlabel('MSEN')
    plt.ylabel('Number of pixels')
    plt.savefig('histogram.png')
    plt.show()

    plt.figure(2)
    plt.imshow(np.reshape(Module, Frame_gt.shape[:-1]))
    plt.colorbar()
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.savefig('error_image.png')
    plt.show()

    return MSEN

def pepn(Frame_gt, Frame_test, th):
    MSEN = []

    Module = np.sqrt(np.add(np.square(np.subtract(Frame_gt[:,:,0],Frame_test[:,:,0])),np.square(np.subtract(Frame_gt[:,:,1],Frame_test[:,:,1]))))

    F_valid_gt = Frame_gt[:, :, 2]

    Module[F_valid_gt == 0] = 0  # 0s in ocluded pixels

    MSEN = np.append(MSEN, Module[F_valid_gt != 0])  # take in account the error of the non-ocluded pixels

    PEPN = (np.sum(MSEN > th) / len(MSEN)) * 100

    return PEPN


def plotFlow(flow, img_path):
	sampling = 10

	img = plt.imread(img_path)

	
	fig1, ax1 = plt.subplots()
	ax1.imshow(img,cmap=plt.get_cmap('gray'))
	plt.gca().invert_yaxis()
	U = flow[:,:,0]
	V = flow[:,:,1]
	valid = flow[:,:,2]
	U = U * valid
	V = V * valid
	X , Y = np.meshgrid(np.arange(0, np.shape(flow)[1]),np.arange(0,np.shape(flow)[0]))
	ax1.set_title("Flow representation frame 45")
	max_vector_length = max(np.max(U),np.max(V))
	Q = ax1.quiver(X[::sampling,::sampling], Y[::sampling,::sampling], U[::sampling,::sampling], V[::sampling,::sampling],
                units='xy', angles='xy', scale = max_vector_length*0.2/sampling, color = "red")
	

	#Q = ax1.quiver(X[::sampling,::sampling], Y[::sampling,::sampling], U[::sampling,::sampling], V[::sampling,::sampling],
    #            units='xy', angles='xy')
	plt.gca().invert_yaxis()
	qk = ax1.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{p}{f}$', labelpos='E',
                   coordinates='figure')
	plt.show()


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep