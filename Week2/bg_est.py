import cv2
import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append("Week1")
import voc_evaluation
import utils_week1, utils_week2
from utils_week1 import read_annotations
from utils_week2 import nms

annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
roi_path = 'data/AICity_data/train/S03/c010/roi.jpg'
result_path = 'week2/output/'


def adaptive_bg_est(image, frame_size, mean, std, alpha, rho):
    alpha = alpha
    rho = rho
    h, w = frame_size
    mask = abs(image - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    segmentation[mask] = 255

    mean = np.where(mask, mean, rho * image + (1 - rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (image - mean) ** 2 + (1 - rho) * std ** 2))

    return segmentation, mean, std, alpha

def morph_filter(img):

    img = cv2.GaussianBlur(img, (5, 5), 10)
    img = cv2.dilate(img, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=6)

    return img


def eval(vidcap, frame_size, mean, std, test_len, alpha, rho):
    annotations_read = read_annotations(annotations_path)
    init_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_id = init_frame
    detections = {}
    annotations = {}
    
    for t in tqdm(range(test_len)):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Frame', frame)    
        segmentation, mean, std, alpha = adaptive_bg_est(frame, frame_size, mean, std, alpha, rho)
        #cv2.imshow('Segmentation', segmentation)
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE) / 255
        segmentation = segmentation * roi
        #cv2.imshow('Segmentation*roi', segmentation)
        segmentation = morph_filter(segmentation).astype(np.uint8)
        #cv2.imshow('Filtered seg', segmentation)

        #save
        #if frame_id >= 1169 and frame_id < 1229 : # if frame_id >= 535 and frame_id < 550
        #    cv2.imwrite(result_path + f"seg_{str(frame_id)}_pp_{str(alpha)}.bmp", segmentation.astype(int))

        output = cv2.connectedComponentsWithStats(segmentation)
        
        frameRes = frame

        bboxes = []
        for i_region in range(len(output[2])):
            if output[2][i_region][4] > 5000 and output[2][i_region][3] < 2000:
                x1 = output[2][i_region][0]
                y1 = output[2][i_region][1]
                x2 = x1 + output[2][i_region][2]
                y2 = y1 + output[2][i_region][3]
                bboxes.append([x1, y1, x2, y2, output[2][i_region][4]])

        frameId = str(frame_id)

        annotations[frameId] = annotations_read[frame_id]
        if len(bboxes) > 0:
            bboxes = np.array(bboxes)
            keep = nms(bboxes, 0.1)
            detections[frameId] = []
            for bbox in bboxes[keep]:
                frameRes = cv2.rectangle(frameRes, (bbox[0], bbox[1]) , (bbox[2], bbox[3]), (0,0,255))
                detections[frameId].append({
                    "bbox": np.array(bbox),
                    "conf": 1.0,
                    "frame": str(frame_id)
                })

        #Show boxes
        #frameRes = cv2.resize(frameRes, (960, 540))
        #cv2.imshow('Frame with boxes', frame)
        #if cv2.waitKey(10) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break

        frame_id += 1

    rec, prec, ap = voc_evaluation.voc_eval(annotations, detections, use_confidence=False)

    return ap


def train(vidcap, frame_size, train_len):
    count = 0
    h, w = frame_size
    #nc = color_space[params['color_space']][1]
    #if nc == 1:
    mean = np.zeros((h, w))
    M2 = np.zeros((h, w))
    #else:
    #    mean = np.zeros((h, w, nc))
    #    M2 = np.zeros((h, w, nc))

    # Compute mean and std
    for t in tqdm(range(train_len)):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.cvtColor(frame, color_space[params['color_space']][0])
        #if params['color_space'] == 'H':
        #    H,S,V = np.split(frame,3,axis=2)
        #    frame=np.squeeze(H)
        #if params['color_space'] == 'L':
        #    L,A,B = np.split(frame,3,axis=2)
        #    frame=np.squeeze(L)
        #if params['color_space'] == 'CbCr':
        #    Y,Cb,Cr= np.split(frame,3,axis=2)
        #    frame=np.dstack((Cb,Cr))
        count += 1
        delta = frame - mean
        mean += delta / count
        delta2 = frame - mean
        M2 += delta * delta2

    mean = mean
    std = np.sqrt(M2 / count)

    print("Mean and std computed")

    #save results
    cv2.imwrite(os.path.join(result_path,"mean_train.png"), mean)
    cv2.imwrite(os.path.join(result_path,"std_train.png"), std)

    return mean, std