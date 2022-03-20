import math
import cv2
import numpy as np
from utils import plot_gaussian_model, nms

frames_folder = '../data/images/'
TOTAL_FRAMES = 2141;
TRAIN_FRAMES = math.ceil(TOTAL_FRAMES * 0.25);
TEST_FRAMES = math.ceil(TOTAL_FRAMES * 0.75);


def classify_frame(u, std, frame, alpha=2.5):
	return (np.abs(frame - u) >= alpha * (std + 2)).astype(np.uint8) * 255

def morph_filter(img):
	kernel = np.ones((3, 3), np.uint8)
	img = cv2.dilate(img, kernel, iterations = 4)
	return cv2.erode(img, kernel, iterations = 2)

if __name__ == '__main__':

	# Load gaussian model
	mean = np.load('../data/gaussian_model_mean.npy')
	std = np.load('../data/gaussian_model_std.npy')

	for i in range(TRAIN_FRAMES, TEST_FRAMES):
		frame = cv2.imread(frames_folder + "%04d.jpeg" % i, cv2.IMREAD_GRAYSCALE)
		classify = classify_frame(mean, std, frame, alpha=5)
		img = morph_filter(classify)

		output = cv2.connectedComponentsWithStats(img)
		frameRes = frame

		bboxes = [];
		for i in range(len(output[2])):
			if output[2][i][4] > 1000:
				x1 = output[2][i][0]
				y1 = output[2][i][1]
				x2 = x1 + output[2][i][2]
				y2 = y1 + output[2][i][3]
				bboxes.append([x1, y1, x2, y2, output[2][i][4]]);

		if len(bboxes) > 0:
			bboxes = np.array(bboxes)
			keep = nms(bboxes, 0.1)
			for bbox in bboxes[keep]:
				#if stats[4] > 1000 and stats[4] < 10000:
				frameRes = cv2.rectangle(frameRes, (bbox[0], bbox[1]) , (bbox[2], bbox[3]), (0,0,255))

		classiRes = cv2.resize(img, (960, 540))
		cv2.imshow('Class', classiRes)
		frameRes = cv2.resize(frameRes, (960, 540))
		cv2.imshow('Frame', frameRes)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
