import math
import cv2
import numpy as np
from utils import plot_gaussian_model

frames_folder = '../data/images/'
TOTAL_FRAMES = 2141;
TRAIN_FRAMES = math.ceil(TOTAL_FRAMES * 0.25);
TEST_FRAMES = math.ceil(TOTAL_FRAMES * 0.75);

def classify_frame(u, std, frame, alpha=2.5):
    return (np.abs(frame - u) >= alpha * (std + 2)).astype(np.uint8) * 255

if __name__ == '__main__':

	# Load gaussian model
	mean = np.load('../data/gaussian_model_mean.npy')
	std = np.load('../data/gaussian_model_std.npy')

	for i in range(TRAIN_FRAMES, TEST_FRAMES):
		frame = cv2.imread(frames_folder + "%04d.jpeg" % i, cv2.IMREAD_GRAYSCALE)
		classify = classify_frame(mean, std, frame, alpha=5)

		kernel = np.ones((4, 4), np.uint8)
		dilation = cv2.dilate(classify, kernel, iterations = 8)
		erosion = cv2.erode(dilation, kernel, iterations = 2)

		output = cv2.connectedComponentsWithStats(erosion)
		frameRes = cv2.resize(frame, (960, 540))
		cv2.imshow('Frame', frameRes)
		for stats in output[2]:
			if stats[4] > 1000 and stats[4] < 10000:
				frameRes = cv2.rectangle(frameRes, (stats[0], stats[1]) , (stats[0]+stats[2], stats[1]+stats[3]), (0,0,255))

		classiRes = cv2.resize(erosion, (960, 540))
		cv2.imshow('Class', classiRes)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
