import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import csv


sys.path.append("Week3")
from utils_week3 import read_detections2

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def create_csv_patches(video_path, gt_path, patches_path, sequence, camera, writer):
    vidcap = cv2.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", num_frames)

    gt = read_detections2(gt_path)

    for frame_id in tqdm(range(num_frames), desc='Creating patches of seq ' + sequence + '/' + camera):
        _, frame = vidcap.read()
        if frame_id in gt:
            gt_bboxes = gt[frame_id]

            for box in gt_bboxes:
                
                crop_img = frame[int(box['bbox'][1]):int(box['bbox'][3]), int(box['bbox'][0]):int(box['bbox'][2])]
                cv2.imwrite(patches_path + f"/{str(box['id'])}_{sequence}_{camera}_{str(frame_id)}.jpg",            #Save gt patches
                             crop_img.astype(int))
                
                filename = str(box['id']) + '_' + sequence + '_' + camera + '_' + str(frame_id) + '.jpg'
                writer.writerow([filename, str(box['id'])])

        frame_id += 1

    return

if __name__ == "__main__":
    
    sequences = ['S03','S04','S01']
    with open('Week5/gt/gt_car_patches_annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["FILENAME", "ID"])  
        
        for seq in sequences:
            videos_path = os.path.join('data/AICity_data/train', seq)
            patches_path =  os.path.join('Week5/gt/gt_car_patches', seq)
            os.makedirs(patches_path, exist_ok=True)
            
            for camera in sorted(mylistdir(videos_path)):
                gt_path = os.path.join(videos_path, camera, 'gt/gt.txt')
                video_path = os.path.join(videos_path, camera, 'vdo.avi')

                print("--------------- VIDEO ---------------")
                print(seq, camera)
                print("--------------------------------------")

                create_csv_patches(video_path, gt_path, patches_path, seq, camera, writer)
