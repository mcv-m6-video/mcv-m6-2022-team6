import utils as ut
import voc_evaluation
import numpy as np
import os
import cv2 as cv
import subprocess as sp
import ffmpeg

os.chdir("..")
# PATHS
annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'
detections_path = {
	'rcnn': 'data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
	'ssd': 'data/AICity_data/train/S03/c010/det/det_ssd512.txt',
	'yolo': 'data/AICity_data/train/S03/c010/det/det_yolo3.txt'
}


def task03 (substractor):

	#choses the substraction method
	if substractor == 'MOG2':
		backSub = cv.createBackgroundSubtractorMOG2()
	elif substractor == 'MOG':
		backSub = cv.bgsegm.createBackgroundSubtractorMOG()
	elif substractor == 'CNT':
		backSub = cv.bgsegm.createBackgroundSubtractorCNT()
	elif substractor == 'GMG':
		backSub = cv.bgsegm.createBackgroundSubtractorGMG()
	elif substractor == 'GSOC':
		backSub = cv.bgsegm.createBackgroundSubtractorGSOC()
	elif substractor == 'LSBP':
		backSub = cv.bgsegm.createBackgroundSubtractorLSBP()
	elif substractor == 'LSBPDesc':
		backSub = cv.bgsegm.createBackgroundSubtractorLSBPDesc()
	elif substractor == 'KNN':
		backSub = cv.createBackgroundSubtractorKNN()

	#read the video
	im_array = []
	video = cv.VideoCapture(video_path)
	if not video.isOpened():
		print('Unable to open the video')

	while True:
		ret, frame = video.read()
		if frame is None:
			break
		fgMask = backSub.apply(frame)
		cv.imshow('FG Mask', fgMask)
		im_array.append(fgMask)
		keyboard = cv.waitKey(30)
		if keyboard == 'q' or keyboard == 27:
			break


if __name__ == '__main__':
	task03('GMG')
	
