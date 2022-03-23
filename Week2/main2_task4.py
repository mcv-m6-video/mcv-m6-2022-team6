import matplotlib
import utils as ut
import voc_evaluation
import numpy as np
import os
import cv2 as cv
import subprocess as sp
import ffmpeg
import pickle
from matplotlib import pyplot as plt
import gc
import glob

os.chdir("..")
# PATHS
annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'
detections_path = {
	'rcnn': 'data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
	'ssd': 'data/AICity_data/train/S03/c010/det/det_ssd512.txt',
	'yolo': 'data/AICity_data/train/S03/c010/det/det_yolo3.txt'
}


def read_frame_colourspace(frame_path, colour_conversion='gray'):
    frame = cv.imread(str(frame_path))  # BGR
    if colour_conversion == 'RGB':
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    elif colour_conversion == 'HSV':
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
		
    elif colour_conversion == 'Luv':
        frame = cv.cvtColor(frame, cv.COLOR_BGR2Luv)
    elif colour_conversion == 'Lab':
        frame = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
    else: 
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    return frame
	
def change_color_map():
	filepath = sorted(glob.glob(os.path.join('data/AICity_data/frames/', 'frame????.png')))
	frame_list_length = 2141
	frames_test = int(frame_list_length)
	
	frame_list = filepath[:frames_test]
	i = 0
	for frame_path in frame_list:
		frame = read_frame_colourspace(frame_path,'HSV')
		cv.imwrite('data/AICity_data/HSV/frame{:04d}.png'.format(i),frame)
		i +=1
	

def get_frames():
	video = cv.VideoCapture(video_path)
	i = 0
	while video.isOpened():
		ret, frame = video.read()
		if ret == False:
			break
		cv.imwrite('data/AICity_data/frames/frame{:04d}.png'.format(i),frame)
		i+=1
def background_substractor(substractor):
	if substractor == 'MOG2':
		backSub = cv.createBackgroundSubtractorMOG2(detectShadows=True)
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
	
	
	return backSub

def detect_objects(frame):
	detections =  []
	ret, label, stats, cent = cv.connectedComponentsWithStats(frame)
	for i in range(len(stats)):
		if stats[i][4] > 100 and stats[i][4] < 500000:
			detections.append([stats[i][0],stats[i][1],stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]])

	return detections

def get_frame_bounding_box(detections, frame):


    frame_detections = []
    for j in range(len(detections)):
        if detections[j][0] == frame:
            frame_detections.append(
                [detections[j][2], detections[j][3], detections[j][4],
                 detections[j][5]])

    return frame_detections


#From team 4 of 2019 Jonathan Poveda and Ferran Perez
def compute_iou(gtExtractor, detections, threshold, frames):

    IoUvsFrames = []
    detections_IoU = []
    TP = 0
    FP = 0
    FN = 0

    for i in range(frames):

        IoUvsFrame = []

        frame_gt = get_frame_bounding_box(gtExtractor.gt, i)

        frame_detections = get_frame_bounding_box(detections, i)

        gt_detections = []

        for y in range(len(frame_detections)):

            maxIoU = 0
            gtDetected = -1

            for x in range(len(frame_gt)):
                iou = ut.get_rect_iou(frame_gt[x], frame_detections[y])
                if iou >= maxIoU:
                    maxIoU = iou
                    gtDetected = x

            detections_IoU.append(maxIoU)

            if maxIoU > threshold:
                TP = TP + 1
                gt_detections.append(gtDetected)
            else:
                FP = FP + 1

            IoUvsFrame.append(maxIoU)

        if not IoUvsFrame:
            IoUvsFrame = [0]

        IoUvsFrames.append(sum(IoUvsFrame) / len(IoUvsFrame))

        for x in range(len(frame_gt)):
            if not gt_detections.__contains__(x):
                FN = FN + 1



    plt.figure(2)
    plt.plot(IoUvsFrames)
    plt.ylabel('IoU')
    plt.xlabel('Frames')
    plt.title('IoU')
    plt.savefig('figure_IoU.png')
    plt.clf()
    plt.cla()
    plt.close()
    # plt.show()

def process_image(image):


    ret, image = cv.threshold(image, 130, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    roi_path = 'data/AICity_data/train/S03/c010/roi.jpg'
    roi = cv.imread(str(roi_path), 0)
    image = cv.bitwise_and(image, roi)

    return image

def analyze_sequence(method, colorMap):
	backSub = background_substractor(method)
	video_path = str(colorMap) + str(method)+'.avi'

	detection_list = []
	
	
	annotations = ut.read_annotations(annotations_path)
	# detect objects with new background 
	#objects
	for i in range(2141):
		frame_path = glob.glob(os.path.join('data/AICity_data/frames/','frame{:04d}.png'.format(i)))
		image = read_frame_colourspace(frame_path[0],colorMap)

		image = backSub.apply(image)
		image = process_image(image)

		# se debe añadir los metodos de deteccion
		detections = detect_objects(image)

		for detection in detections:
			detection_list.append([i, 0, detection[0], detection[1], detection[2],detection[3], 1])

		image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
		cv.imshow("Frame", image)
		cv.waitKey(1)
		
	iou_frame = ut.get_frame_iou(annotations, detection_list, 2141)
	print(iou_frame)




if __name__ == '__main__':
	#task03('MOG2')
	#get_frames()
	analyze_sequence('MOG2', 'HSV')
	
