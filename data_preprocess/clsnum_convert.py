import os
import numpy as np

convert = {"0":"0", "1":"0", "2":"0", "3":"1"}

train_f = open("loveu_wide_train_2s_30fps_annotation_valid.txt")
val_f   = open("loveu_wide_val_2s_30fps_annotation_valid.txt")
train_f_w = open("loveu_wide_train_2cls_2s_30fps_annotation_valid.txt", 'w')
val_f_w   = open("loveu_wide_val_2cls_2s_30fps_annotation_valid.txt", 'w')

for line in train_f.readlines():
    path, label = line.strip().split(" ")
    train_f_w.write(path + " " + convert[label] + "\n")

for line in val_f.readlines():
    path, label = line.strip().split(" ")
    val_f_w.write(path + " " + convert[label] + "\n")

