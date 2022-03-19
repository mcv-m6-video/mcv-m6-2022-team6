import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

frames_folder = '../data/images/'
TOTAL_FRAMES = 2141;
TRAIN_FRAMES = math.ceil(TOTAL_FRAMES * 0.25);
TEST_FRAMES = math.ceil(TOTAL_FRAMES * 0.75);

def generate_u():
	u = np.zeros((1080, 1920))
	for i in range(1, TRAIN_FRAMES):
		frame = cv2.imread(frames_folder + "%04d.jpeg" % i, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255;
		u = u + frame

	u /= TRAIN_FRAMES
	return u

def generate_sigma(u):
	sigma = np.zeros((1080, 1920))
	for i in range(1, TRAIN_FRAMES):
		frame = cv2.imread(frames_folder + "%04d.jpeg" % i, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255;
		sigma = sigma + (frame - u)

	sigma /= TRAIN_FRAMES
	return sigma

def classify_frame(u, sigma, frame, alpha=2.5):
	return (np.abs(frame - u) >= alpha * (sigma + 2)).astype(np.float32)

if __name__ == '__main__':

	print("Generating u")
	u = generate_u()

	print("Generating sigma")
	sigma = generate_sigma(u)

	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1)
	imgplot = plt.imshow(u, cmap='gray')
	ax.set_title('Mean')
	ax = fig.add_subplot(1, 2, 2)
	imgplot = plt.imshow(sigma, cmap='gray')
	imgplot.set_clim(0.0, 0.7)
	ax.set_title('Sigma')
	plt.show()

	for i in range(TRAIN_FRAMES, TEST_FRAMES):
		frame = cv2.imread(frames_folder + "%04d.jpeg" % i, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255;
		classify = classify_frame(u, sigma, frame)

		frameRes = cv2.resize(frame, (960, 540))
		classiRes = cv2.resize(classify, (960, 540))
		cv2.imshow('Frame', frameRes)
		cv2.imshow('Class', classiRes)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break



