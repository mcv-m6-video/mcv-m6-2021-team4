import json
import numpy as np
import matplotlib.pyplot as plt
import sys

experiment_folder = sys.argv[1]

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

plt.plot(x1,y1, color="blue", label="Train Loss")

validation_loss= {}
flag_val = False
for x in experiment_metrics:
    if flag_val:
        if 'validation_loss' in x:
            validation_loss[x['iteration']] = x['validation_loss']

    if 'validation_loss' in x:
        flag_val = True

x2=[]
y2=[]
for k, v in validation_loss.items():
    x2.append(k)
    y2.append(np.mean(np.array(v)))

plt.plot(x2, y2, color="orange", label="Val Loss")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tick_params(axis='y')
plt.title('Faster R-CNN: Train and val loss with LR=1e-5')
plt.legend(loc='upper right')


plt.savefig(experiment_folder+'loss_curves.png')
