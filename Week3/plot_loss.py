import json
import numpy as np
import matplotlib.pyplot as plt

experiment_folder = 'C:/Users/servi/Desktop/Computer Vision Master/M6. Video Analysis/Project/Week3/faster_rcnn_R_50_FPN_3x/lr_0_001_iter_2000_batch_32/' #Change

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + 'metrics.json')

train_loss = {}
for x in experiment_metrics:
    if 'total_loss' in x:
            train_loss[x['iteration']] = x['total_loss']

x1=[]
y1=[]
for k, v in train_loss.items():
    x1.append(k)
    y1.append(np.mean(np.array(v)))

plt.plot(x1,y1, color="blue", label="Faster R-CNN")

val_loss = {}
for x in experiment_metrics:
    if 'total_val_loss' in x:
            val_loss[x['iteration']] = x['total_val_loss']

x2=[]
y2=[]
for k, v in val_loss.items():
    x2.append(k)
    y2.append(np.mean(np.array(v)))

plt.plot(x2,y2, color="orange", label="Faster R-CNN")


plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tick_params(axis='y')
plt.title('Faster R-CNN: Train and val loss with LR=1e-5') #change
plt.legend(loc='upper right')


plt.savefig(experiment_folder+'loss_curves.png')