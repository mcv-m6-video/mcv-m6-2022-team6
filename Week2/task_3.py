
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
sys.path.append('Week1')
import utils_week1 as utils_week1
import utils_week2 as utils_week2


# PATHS
sota_method = 'GSOC'
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'
TRAIN_PERCENTAGE = 0.75
SOTA_METHOD_SUPPORTED = ['MOG','MOG2','LSBP','KNN','GSOC']
HISTORY  = 100
if __name__ == '__main__':
    vidcap = cv2.VideoCapture(video_path)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Found a total of {} frames".format(frame_count))
    train_idx = int(TRAIN_PERCENTAGE*frame_count)
    test_idx = frame_count-train_idx
    print('training with a total of {} and testing with {} frames'.format(train_idx,test_idx))

    if sota_method not in SOTA_METHOD_SUPPORTED:
        print('Need to choose SOTA method between {}'.format(SOTA_METHOD_SUPPORTED))
        sys.exit()
    else:
        pass
    if sota_method == 'KNN':
        bgsem = cv2.createBackgroundSubtractorKNN(history=HISTORY, detectShadows=True)
    elif sota_method == 'MOG':
        bgsem = cv2.bgsegm.createBackgroundSubtractorMOG(history=HISTORY, nmixtures=2, backgroundRatio=0.7)
    elif sota_method == 'LSBP':
        bgsem = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif sota_method == 'MOG2':
        bgsem = cv2.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=36, detectShadows=True)    
    elif sota_method == 'GSOC':
        bgsem = cv2.bgsegm.createBackgroundSubtractorGSOC()

    bgsem = utils_week2.train_model(vidcap, train_idx, bgsem)
    ap = utils_week2.eval_sota(vidcap, test_idx, bgsem)
    print('obtained ap equal to {} for method {}'.format(ap,sota_method))