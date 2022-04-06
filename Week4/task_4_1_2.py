import argparse
import sys
import cv2
from cv2 import FarnebackOpticalFlow
from pyoptflow.pyoptflow import HornSchunck
from pyoptflow.pyoptflow.plots import compareGraphs
sys.path.append('./')
from Week1 import flow_read, msen, pepn, plotFlow
import time
import numpy as np
from matplotlib import pyplot as plt

from flow_block_matching import get_optical_flow


showPlots = True

def Farneback(img1, img2):
    #Let's help the algorithm to work better working in a easly work space
    im1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #No let's pass  to the Farneback algorighm both images (just hue channel)
    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5,3,15,3,5,1.2,0)
    #Change from cartagonal coords to polar to represent it. But keep the cartagonal to compare it with other methods
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    hsv_rep = np.zeros_like(img1)
    hsv_rep [...,1] = 255
    hsv_rep[...,0] = ang*180/np.pi/2
    hsv_rep[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv_rep,cv2.COLOR_HSV2BGR)
    if showPlots:
        cv2.imshow('Farneback_representation', bgr)
    cv2.imwrite('Farneback_flow_representation.png',bgr)

    return flow

def Lucas_kanade(img1, img2):

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    old_frame = img1

    frame = img2

    flow = np.zeros((old_frame.shape[0], old_frame.shape[1], 2))

    flow_1 = np.zeros((old_frame.shape[0], old_frame.shape[1], 3))

    u = np.zeros([old_frame.shape[0], old_frame.shape[1]])
    v = np.zeros([old_frame.shape[0], old_frame.shape[1]])

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s = time.time()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    e = time.time()

    print('Time Taken: %.2f seconds' % (e - s))
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    dpi = 80
    im = np.array(img1)
    height = im.shape[1]
    width = im.shape[0]
    fig = plt.figure(figsize=(height / dpi, width / dpi), dpi=dpi)

    colors = 'rgb'
    c = colors[2]

    for idx, good_point in enumerate(good_old):
        old_gray_x = good_point[1]
        old_gray_y = good_point[0]
        frame_gray_x = good_new[idx][1]
        frame_gray_y = good_new[idx][0]

        flow_1[int(old_gray_x), int(old_gray_y)] = np.array([frame_gray_x - old_gray_x, frame_gray_y - old_gray_y, 1])
        u[int(old_gray_x), int(old_gray_y)] = np.array([(frame_gray_x - old_gray_x)])
        v[int(old_gray_x), int(old_gray_y)] = np.array([(frame_gray_y - old_gray_y)])
        plt.arrow(old_gray_y, old_gray_x, int(frame_gray_y)*0.1, int(frame_gray_x)*0.1, head_width=5, head_length=5, color=c)

    if showPlots:
        plt.imshow(im,cmap='gray')
        plt.axis('off')
        plt.show()
    cv2.imwrite("lucas_flow.png", im)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    flow_1 = np.ndarray((u.shape[0], u.shape[1], 3))
    flow_1[:, :, 0] = u
    flow_1[:, :, 1] = v
    flow_1[:, :, 2] = np.ones((u.shape[0], u.shape[1]))

    return flow, flow_1

def horn_schunck(img1, img2):

    im1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    s = time.time()
    U, V = HornSchunck(im1, im2, alpha=1.0, Niter=100)
    e = time.time()
    print('Time Taken: %.2f seconds' % (e - s))

    flow = np.concatenate((U[..., None], V[..., None]), axis=2)

    flow_1 = np.ndarray((U.shape[0], U.shape[1], 3))
    flow_1[:, :, 0] = U
    flow_1[:, :, 1] = V
    flow_1[:, :, 2] = np.ones((U.shape[0], U.shape[1]))

    if showPlots:
        #compareGraphs(U, V, im2)
        hsv = np.zeros_like(img1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('horn_schunck_output', bgr)
        cv2.imwrite("horn_flow.png", bgr)

    return flow, flow_1



def task_4_1_2():
    image_number = 157
    img1 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_10.png".format(image_number))
    img2 = cv2.imread("data/results_opticalflow_kitti/original/{:0>6}_11.png".format(image_number))
    F_gt = flow_read("data/results_opticalflow_kitti/groundtruth/{:0>6}_10.png".format(image_number))
    start_time = time.time()
    method = "shunck"
    if  method == "Farneback":
        result = Farneback(img1, img2)
    elif method == "shunck":
        result , result2 = horn_schunck(img1,img2)
    else:
        result, results2 = Lucas_kanade(img1, img2)
    flow_time = time.time() - start_time
    print("--- %s seconds ---" % (flow_time))
    MSEN = msen(F_gt, result)
    PEPN = pepn(F_gt, result, 3)
    print('MSEN:', MSEN)
    print('PEPN:', PEPN)
    print("task done")



if __name__ == '__main__':
    task_4_1_2()