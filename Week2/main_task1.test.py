import math
import cv2
import numpy as np
from utils import plot_gaussian_model

frames_folder = '../data/images/'
TOTAL_FRAMES = 2141;
TRAIN_FRAMES = math.ceil(TOTAL_FRAMES * 0.25);
TEST_FRAMES = math.ceil(TOTAL_FRAMES * 0.75);

def classify_frame(u, std, frame, alpha=2.5):
	return (frame - u >= alpha * (std + 2)).astype(np.float32)

if __name__ == '__main__':

	# Load gaussian model
	mean = np.load('../data/gaussian_model_mean.npy')
	std = np.load('../data/gaussian_model_std.npy')

	out = cv2.VideoWriter('../data/outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1920, 1080))
	for i in range(TRAIN_FRAMES, TEST_FRAMES):
		frame = cv2.imread(frames_folder + "%04d.jpeg" % i, cv2.IMREAD_GRAYSCALE);
		classify = classify_frame(mean, std, frame, alpha=0.2)

		# frameRes = cv2.resize(frame, (960, 540))
		# cv2.imshow('Frame', frameRes)

		classiRes = cv2.resize(classify, (960, 540))
		out.write(classify.astype(np.uint8))
		cv2.imshow('Class', classiRes)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

	cv2.destroyAllWindows()
	out.release()
