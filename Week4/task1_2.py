import cv2
import time
from lib import pyflow

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

im1 = cv2.imread("../data/results_opticalflow_kitti/original/000045_10.png")
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)/255.;
im2 = cv2.imread("../data/results_opticalflow_kitti/original/000045_11.png")
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)/225.;

s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                                  nSORIterations, colType)
e = time.time()
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
