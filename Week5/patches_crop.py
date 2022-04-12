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

def create_csv_patches(video_path, det_path, patches_path, sequence, camera, writer):
    vidcap = cv2.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", num_frames)

    det = read_detections2(det_path)

    for frame_id in tqdm(range(num_frames), desc='Creating patches of seq ' + sequence + '/' + camera):
        _, frame = vidcap.read()
        if frame_id in det:
            det_bboxes = det[frame_id]

            for box in det_bboxes:
                crop_img = frame[int(box['bbox'][1]):int(box['bbox'][3]), int(box['bbox'][0]):int(box['bbox'][2])]
                cv2.imwrite(patches_path + f"/{str(box['id'])}_{sequence}_{camera}_{str(frame_id)}.jpg",
                             crop_img.astype(int))
                center=[(box['bbox'][0] + box['bbox'][2]) // 2, (box['bbox'][1] + box['bbox'][3]) // 2]
                filename = str(box['id']) + '_' + sequence + '_' + camera + '_' + str(frame_id) + '.jpg'
                writer.writerow([filename, str(box['id']), sequence, camera, str(frame_id), str(box['bbox'][0]), str(box['bbox'][1]),
                                 str(box['bbox'][2]), str(box['bbox'][3]), str(center[0]), str(center[1])])

        frame_id += 1

    return

if __name__ == "__main__":
    sequence = 'S03'
    videos_path = os.path.join('data/AICity_data/train', sequence)
    detections_path = os.path.join('data/detections/retina', sequence)
    patches_path =  os.path.join('patches', sequence)
    os.makedirs(patches_path, exist_ok=True)

    with open('car_patches_annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["FILENAME", "ID", "SEQUENCE", "CAMERA", "FRAME", "XTL", "YTL", "XBR", "YBR", "CENTER_X", "CENTER_Y"])  # header

        for camera in sorted(mylistdir(detections_path)):
            det_path = os.path.join(detections_path, camera, 'overlap_filtered_detections.txt')
            video_path = os.path.join(videos_path, camera, 'vdo.avi')

            print("--------------- VIDEO ---------------")
            print(sequence, camera)
            print("--------------------------------------")

            create_csv_patches(video_path, det_path, patches_path, sequence, camera, writer)
