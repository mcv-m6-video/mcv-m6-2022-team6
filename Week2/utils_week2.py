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


def get_frames(video, frame=0):
	return read_video(video)[0]


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