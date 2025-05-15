import os
import numpy as np
from tools.infer import run
import csv


# 清空data_all/results文件夹
for root, dirs, all_files in os.walk("results/"):
    for name in all_files:
        os.remove(os.path.join(root, name))
# 获取data_all文件夹下的所有jpg图片文件
path = "data/"
files = []
for root, dirs, all_files in os.walk(path):
    for name in all_files:
        if (name.endswith(".jpg") or name.endswith(".JPG")) and name.split(".")[0] != "0":
            files.append(os.path.join(root, name))
# 运行tools下的infer.py
run(weights='model/best_ckpt.pt',
    source='data/',
    save_txt=True,
    yaml='data/dataset.yaml',
    not_save_img=False,
    save_dir='results/',
    img_size=320,
    conf_thres=0.1,
    iou_thres=0.002,
    max_det=5000,
    view_img=False,
    classes=None,
    agnostic_nms=False,
    project='data/',
    name='exp6',
    hide_labels=True,
    hide_conf=False,
    half=False,
    device='0',
    )
# 读取结果文件夹下的所有txt文件，获取每个txt的文本行数，以及每行的长度
path = 'results/labels/'
file_list = os.listdir(path)
num_list = []
len_list = []
name_list = []
file_list.sort()

for file in file_list:
    count = 0
    length_line = []
    with open(path + file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(' ')
            length = np.sqrt((float(line[3])) ** 2 + (float(line[4])) ** 2)*10
            if length < 10:
                length_line.append(length)
                count += 1
    len_list.append(np.mean(length_line))
    num_list.append(count)
    name_list.append(file.split(".")[0])

# 将结果写入csv文件
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['File Name', 'Num', 'Length'])
    for i in range(len(file_list)):
        writer.writerow([name_list[i], num_list[i], len_list[i]])
print("done")
