import math
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

from utils_week2 import nms, plot_gaussian_model
from Week1.voc_evaluation import voc_eval
import Week1.utils_week1 as ut1

frames_folder = 'data/images/'
annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
TOTAL_FRAMES = 2141
TRAIN_FRAMES = math.ceil(TOTAL_FRAMES * 0.25)
TEST_FRAMES = math.ceil(TOTAL_FRAMES * 0.75)
SHOW_VIDEO = True


def classify_frame(u, std, frame, alpha=2.5):
	return (np.abs(frame - u) >= alpha * (std + 2)).astype(np.uint8) * 255

def morph_filter(img, roi):
	img = img * roi;
	img = cv2.GaussianBlur(img, (5, 5), 10)
	img = cv2.dilate(img, np.ones((3, 3)))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=6)
	return img

if __name__ == '__main__':

	alphas = [1, 3, 11]
	for alpha in alphas:
		#writer = imageio.get_writer("Week2/results/alpha_cars%02d.gif" % alpha, mode="I")

		# Load gaussian model
		mean = np.load('data/gaussian_model_mean.npy')
		std = np.load('data/gaussian_model_std.npy')
		#plot_gaussian_model(mean, std)

		annotations_read = ut1.read_annotations(annotations_path)
		roi = cv2.imread('data/roi2.jpeg', cv2.IMREAD_GRAYSCALE);
		roi = (roi > 125).astype(np.uint8)

		detections = {}
		annotations = {}
		frameIds = []
		frameMeanZone = []
		for i in range(TRAIN_FRAMES, TOTAL_FRAMES):
			frame = cv2.imread(frames_folder + "%04d.jpeg" % i, cv2.IMREAD_GRAYSCALE)
			classify = classify_frame(mean, std, frame, alpha=alpha)
			img = morph_filter(classify, roi)
			output = cv2.connectedComponentsWithStats(img)
			frameRes = frame

			bboxes = []
			for i_region in range(len(output[2])):
				if output[2][i_region][4] > 5000 and output[2][i_region][3] < 2000:
					x1 = output[2][i_region][0]
					y1 = output[2][i_region][1]
					x2 = x1 + output[2][i_region][2]
					y2 = y1 + output[2][i_region][3]
					bboxes.append([x1, y1, x2, y2, output[2][i_region][4]])

			frameId = str(i - 1);

			annotations[frameId] = annotations_read[i - 1];
			if len(bboxes) > 0:
				bboxes = np.array(bboxes)
				keep = nms(bboxes, 0.1)
				detections[frameId] = []
				for bbox in bboxes[keep]:
					frameRes = cv2.rectangle(frameRes, (bbox[0], bbox[1]) , (bbox[2], bbox[3]), (0,0,255))
					detections[frameId].append({
						"bbox": np.array(bbox),
						"conf": 1.0,
						"frame": str(i - 1)
					});

			if SHOW_VIDEO:
				classiRes = cv2.resize(img, (960, 540))
				cv2.imshow('Class', classiRes)
				frameRes = cv2.resize(frameRes, (960, 540))
				cv2.imshow('Frame', frameRes)
				#writer.append_data(frameRes)
				if cv2.waitKey(10) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break

		#writer.close()
		rec, prec, ap = voc_eval(annotations, detections)
		print("Alpha %d" % alpha)
		print("mAP: " + str(ap * 100))
		print("Rec: " + str(np.mean(rec)))
		print("Prec: " + str(np.mean(prec)))