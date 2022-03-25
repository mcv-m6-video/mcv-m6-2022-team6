import cv2
import numpy as np
from tqdm import tqdm
import sys
import itertools
from bg_est import train, eval
import matplotlib.pyplot as plt
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'

vidcap = cv2.VideoCapture(video_path)  #Reading input file
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))    #number of total frames
frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))    #width frame
frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  #height frame
frame_size = [frame_height,frame_width]                    #frame size 
print("Total frames: ", frame_count)

train_len = int(0.25 * frame_count) #First 25% frames for training
test_len = frame_count - train_len  #Second 75% left background adapts
print("Train frames: ", train_len)
print("Test frames: ", test_len)

# Train
mean, std = train(vidcap, frame_size, train_len)

# Evaluate
#Grid-search
alpha = 3 #np.arange(3, 8, 1).tolist() 
rho = 0.02 #np.arange(0.001, 0.01, 0.004).tolist()  

#all_combinations = list(itertools.product(alpha, rho))
#print(all_combinations)
#print("Num. of comb.", len(all_combinations))

ap = []
#i=0
#for alpha, rho in all_combinations:
    #if i>0:
    #    vidcap = cv2.VideoCapture(video_path)
ap.append(eval(vidcap, frame_size, mean, std, test_len, alpha, rho))
    #print(i)
print(ap)


#max_position = ap.index(max(ap))
#print('The maximum accuracy is obtained with a value of {} with alpha {} and rho {}'.format(np.round(ap[max_position],5),all_combinations[max_position][0],all_combinations[max_position][1]))

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter([combi[0] for combi in all_combinations], [combi[1] for combi in all_combinations], ap)
#ax.set_xlabel('alpha')
#ax.set_ylabel('rho')
#ax.set_zlabel('mAP')
#plt.show()