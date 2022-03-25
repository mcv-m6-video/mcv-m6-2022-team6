import cv2
import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append("Week1")
import voc_evaluation
import utils_week1, utils_week2
from utils_week1 import read_annotations
from utils_week2 import nms
import imageio


annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
roi_path = 'data/AICity_data/train/S03/c010/roi.jpg'
result_path = 'Week2/output/'


def adaptive_bg_est(image, frame_size, mean, std, alpha, rho): #adaptive modelling

    h, w = frame_size
    mask = abs(image - mean) >= alpha * (std + 2)  #foreground pixels (Bool: true or false)

    segmentation = np.zeros((h, w)) #black "image"
    segmentation[mask] = 255  #gives white color to those foreground pixels

    #Adaptive modelling
    mean = np.where(mask, mean, rho * image + (1 - rho) * mean)  #np.where(condition[,x,y]) -> when True, yield x(foreground), otherwise y(background)
    std = np.where(mask, std, np.sqrt(rho * (image - mean) ** 2 + (1 - rho) * std ** 2))

    return segmentation, mean, std, alpha

def morph_filter(seg):

    kernel = np.ones((2, 2), np.uint8)
    seg = cv2.erode(seg, kernel, iterations=1)
    kernel = np.ones((3,4), np.uint8)
    seg = cv2.dilate(seg, kernel, iterations=1)

    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((7, 4), np.uint8))
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((4, 7), np.uint8))

    return seg


def fg_bboxes(seg, frame_id):
    bboxes = {}
    roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE) / 255
    segmentation = seg * roi  #convert to a binary image to use findContours()
    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(segmentation, contours,-1,(0,0,255),3)
    #cv2.imshow("img", segmentation)
    
    frameId = str(frame_id)
    idx = 0
    for c in contours:
        rect = cv2.boundingRect(c)  
        if rect[2] < 50 or rect[3] < 50 or rect[2]/rect[3] < 0.8:  # rect[2]= width, rect[3]=height
            continue  # Discard small contours

        x, y, w, h = rect #return: top left corner coord (xlt,ylt), width and height of the rectangle
        bboxes[idx] = []
        bboxes[idx].append({
                    "bbox": np.array([float(x), float(y), float(x) + float(w), float(y) + float(h)]),
                    "conf": None,
                    "difficult": False,
                    "frame": frame_id
                })

        idx += 1

    return bboxes

def eval(vidcap, frame_size, mean, std, test_len, alpha, rho):
    annotations_read = read_annotations(annotations_path)
    init_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))  #get frame position (535)
    frame_id = init_frame  #frame number
    
    detections = {}
    annotations = {}
    idx=0
    #writer = imageio.get_writer("Week2/bboxes.gif", mode="I") #gift
    for t in tqdm(range(test_len)):
        
        _, frame = vidcap.read() #read a frame
        #cv2.imshow('Colored Frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Grayscaled Frame', frame) # Display the resulting frame
          
        segmentation, mean, std, alpha = adaptive_bg_est(frame, frame_size, mean, std, alpha, rho) #apply adaptive modelling
        #cv2.imshow('Segmentation', segmentation)
        
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE) / 255  #region of interest (0's and 1's)
        segmentation = segmentation * roi  #select foreground pixels of region of interest (remove noise)
        #cv2.imshow('Segmentation of region of interest', segmentation)
        
        segmentation = morph_filter(segmentation).astype(np.uint8) #apply morphological filter
        #cv2.imshow('Filtered segmentation', segmentation)

        #Detect foreground bounding boxes of the frame
        det_bboxes = fg_bboxes(segmentation, frame_id)
        
        #Detections dict
        for det in det_bboxes:
            detections[idx] = det_bboxes[det]
            idx = idx+1
        
        #annotations dict
        gt_bboxes = []
        if frame_id in annotations_read:
            gt_bboxes = annotations_read[frame_id]
        annotations[frame_id] = gt_bboxes

        #annotations[frameId] = annotations_read[frame_id]
    

        #Show boxes
        #frameRes = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #frameRes = cv2.resize(frameRes, (960, 540))
        #cv2.imshow('Frame with boxes', frameRes)
        #if cv2.waitKey(10) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break

        #writer.append_data(seg_boxes.astype(np.uint8))
        frame_id += 1

    rec, prec, ap = voc_evaluation.voc_eval(annotations, detections, use_confidence=False)

    return ap


def train(vidcap, frame_size, train_len):
    count = 0
    h, w = frame_size
    
    mean = np.zeros((h, w))
    M2 = np.zeros((h, w))

    # Compute mean and std for training frames
    for t in tqdm(range(train_len)):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert frame to grayscale
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