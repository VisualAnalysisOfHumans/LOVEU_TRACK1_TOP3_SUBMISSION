import os
import cv2
import pickle
import pandas as pd
import numpy as np
import json

labels = ['ShotChangeGradualRange:', 'EventChange', '']
cn = [0, 0, 0, 0]
data_root = "/home/notebook/data/personal/loveu/data/valset/"
data_target = "/home/notebook/data/personal/loveu/data/loveu_wide_val_1s_30fps/"
anno_file = open("loveu_wide_val_1s_30fps_annotation.txt", 'w')
anno_cmds = open("loveu_wide_val_1s_30fps_cmds.txt", 'w')
cmds = []
suffixs = [".mkv", ".mp4", ".mp4.mkv", ".mp4.webm"]

vid2bias = {}
with open("validate.csv") as f:
    for line in f:
        line = line.strip().split(",")
        vid = line[1]
        bias = int(line[2])
        vid2bias[vid] = bias

# Generate frameidx for shot/event change
def generate_frameidx_from_raw(min_change_duration=0.3, split='valnew'):
    assert split in ['train','val','valnew','test']

    with open('eval/k400_{}_raw_annotation.pkl'.format(split),'rb') as f:
        dict_raw = pickle.load(f, encoding='lartin1')
    
    mr345 = {}     
    cnf = 0
    for filename in dict_raw.keys():
        cnf += 1
        ann_of_this_file = dict_raw[filename]['substages_timestamps']

        vid = dict_raw[filename]['path_video'].split("/")[-1][:11]
        cls = dict_raw[filename]['path_video'].split("/")[0] + "/"
        file_path = None
        for suffix in suffixs:
            tmp_path = data_root + cls + vid + suffix
            if os.path.exists(tmp_path):
                file_path = tmp_path
                break

        if file_path == None:
            print("not exists", data_root + cls + vid)
            continue

        if not (len(ann_of_this_file) >= 3):
            continue
            
        try:
            fps = dict_raw[filename]['fps']
            num_frames = int(dict_raw[filename]['num_frames'])
            video_duration = dict_raw[filename]['video_duration']
            avg_f1 = dict_raw[filename]['f1_consis_avg']
        except:
            continue
        
        mr345[filename] = {}
        mr345[filename]['num_frames'] = int(dict_raw[filename]['num_frames'])
        mr345[filename]['path_video'] =dict_raw[filename]['path_video']
        mr345[filename]['fps'] = dict_raw[filename]['fps']
        mr345[filename]['video_duration'] = dict_raw[filename]['video_duration']
        mr345[filename]['path_frame'] = dict_raw[filename]['path_video'].split('.mp4')[0]
        mr345[filename]['f1_consis'] = []
        mr345[filename]['f1_consis_avg'] = avg_f1
        
        mr345[filename]['substages_myframeidx'] = []
        mr345[filename]['substages_timestamps'] = []
        mr345[filename]['substages_changetype'] = []

        mr345[filename]['substages_timestamps_all'] = []
        mr345[filename]['substages_changetype_all'] = []
        for ann_idx in range(len(ann_of_this_file)):
            # remove changes at the beginning and end of the video; 
            ann = ann_of_this_file[ann_idx]
            tmp_ann = []
            change_shot_range_start = []
            change_shot_range_end = []
            change_event = []
            change_shot_timestamp = []
            for p in ann:
                st = p['start_time']
                et = p['end_time']
                l = p['label'].split(' ')[0]
                if l not in labels:
                    labels.append(l)
                if (st+et)/2<min_change_duration or (st+et)/2>(video_duration-min_change_duration): continue
                tmp_ann.append(p)
                if l == 'EventChange':
                    change_event.append((st+et)/2)
                elif l == 'ShotChangeGradualRange:':
                    change_shot_range_start.append(st)
                    change_shot_range_end.append(et)
                else:
                    change_shot_timestamp.append((st+et)/2)
            
            # consolidate duplicated/very close timestamps
            # if two shot range overlap, merge
            i = 0
            while i < len(change_shot_range_start)-1:
                while change_shot_range_end[i]>=change_shot_range_start[i+1]:
                    change_shot_range_start.remove(change_shot_range_start[i+1])
                    if change_shot_range_end[i]<=change_shot_range_end[i+1]:
                        change_shot_range_end.remove(change_shot_range_end[i])
                    else:
                        change_shot_range_end.remove(change_shot_range_end[i+1])
                    if i==len(change_shot_range_start)-1:
                        break
                i+=1      
            
            # if change_event or change_shot_timestamp falls into range of shot range, remove this change_event
            for cg in change_event:
                for i in range(len(change_shot_range_start)):
                    if cg<=(change_shot_range_end[i]+min_change_duration) and cg>=(change_shot_range_start[i]-min_change_duration):
                        change_event.remove(cg)
                        break
            for cg in change_shot_timestamp:
                for i in range(len(change_shot_range_start)):
                    if cg<=(change_shot_range_end[i]+min_change_duration) and cg>=(change_shot_range_start[i]-min_change_duration):
                        change_shot_timestamp.remove(cg)
                        break
            
            # if two timestamp changes are too close, remove the second one between two shot changes, two event changes; shot vs. event, remove event
            change_event.sort()
            change_shot_timestamp.sort()
            tmp_change_shot_timestamp = change_shot_timestamp
            tmp_change_event = change_event
            #"""
            i = 0
            while i <= (len(change_event)-2):
                if (change_event[i+1]-change_event[i])<=2*min_change_duration:
                    tmp_change_event.remove(change_event[i+1])
                else:
                    i += 1
            i = 0
            while i <= (len(change_shot_timestamp)-2):
                if (change_shot_timestamp[i+1]-change_shot_timestamp[i])<=2*min_change_duration:
                    tmp_change_shot_timestamp.remove(change_shot_timestamp[i+1])
                else:
                    i += 1
            for i in range(len(tmp_change_shot_timestamp)-1):
                j = 0
                while j <= (len(tmp_change_event)-1):
                    if abs(tmp_change_shot_timestamp[i]-tmp_change_event[j])<=2*min_change_duration:
                        tmp_change_event.remove(tmp_change_event[j])
                    else:
                        j += 1
            change_shot_timestamp = tmp_change_shot_timestamp
            change_event = tmp_change_event
            change_shot_range = []
            for i in range(len(change_shot_range_start)):
                change_shot_range += [(change_shot_range_start[i]+change_shot_range_end[i])/2]

            change_all = change_event + change_shot_timestamp + change_shot_range
            change_type_all = [0] * len(change_event) + [1] * len(change_shot_timestamp) + [2] * len(change_shot_range)
            
            change_type_all = list(np.array(change_type_all)[np.array(change_all).argsort()])
            change_all.sort()

            time_change_all = change_all       

            change_all = np.floor(np.array(change_all)*fps)
            tmp_change_all = []
            for cg in change_all:
                tmp_change_all += [min(num_frames-1, cg)]

            mr345[filename]['substages_myframeidx'] += [tmp_change_all]
            mr345[filename]['substages_timestamps'] += [time_change_all]
            mr345[filename]['substages_changetype'] += [change_type_all]
            mr345[filename]['substages_timestamps_all'] += time_change_all
            mr345[filename]['substages_changetype_all'] += change_type_all
            mr345[filename]['f1_consis'] += [dict_raw[filename]['f1_consis'][ann_idx]]
        
        mr345[filename]['substages_changetype_all'] = list(np.array(mr345[filename]['substages_changetype_all'])[np.array(mr345[filename]['substages_timestamps_all']).argsort()])
        mr345[filename]['substages_timestamps_all'].sort()

        timestamp_last = 0
        ms_end = mr345[filename]['video_duration'] * 1000 # ms
        
        if vid in vid2bias.keys() and vid2bias[vid] == 0:
            timestamp_bias = 0 # -1s to 11s in val
        else:
            timestamp_bias = 1000
        gap_neg = 1000 # 0.6s
        stride_neg = 200
        gap_pos = 300 # 0.1s
        for i in range(len(mr345[filename]['substages_timestamps_all'])):
            timestamp = int(mr345[filename]['substages_timestamps_all'][i] * 1000)
            label = mr345[filename]['substages_changetype_all'][i]
            if timestamp - timestamp_last > gap_neg:
                this_time = timestamp_last
                for i in range(100): # add 1000ms at a time
                    this_time_beg = this_time + i * stride_neg
                    this_time_end = this_time_beg + gap_neg
                    if this_time_end > timestamp:
                        break
                    cmd = "ffmpeg -ss " + str(max(timestamp_bias, (this_time_beg + timestamp_bias))) + "ms -i \"" + file_path + "\" -crf 10 -an -vf \"fps=30, scale=-2:256\"" + \
                                " -to " + str(gap_neg) + "ms " + data_target + filename + "_s" + str(this_time_beg) + "_t" + str(gap_neg) + "_neg_3.mp4"
                    anno_file.write(data_target + filename + "_s" + str(this_time_beg) + "_t" + str(gap_neg) + "_neg_3.mp4 3\n")
                    anno_cmds.write(cmd+"\n")
                    cmds.append(cmd)
                    cn[3] += 1
            if timestamp - timestamp_last < gap_pos:
                continue
            cmd = "ffmpeg -ss " + str(max(timestamp_bias, (timestamp - 500 + timestamp_bias))) + "ms -i \"" + file_path + "\" -crf 10 -an -vf \"fps=30, scale=-2:256\"" + \
                                " -to " + str(1000) + "ms " + data_target + filename + "_s" + str(timestamp) + "_t1000_pos_" + str(label) + ".mp4"
            anno_file.write(data_target + filename + "_s" + str(timestamp) + "_t1000_pos_" + str(label) + ".mp4 " + str(label) + "\n")
            anno_cmds.write(cmd+"\n")
            cmds.append(cmd)
            cn[label] += 1
            timestamp_last = timestamp
        if ms_end - timestamp_last > gap_neg:
            this_time = timestamp_last
            for i in range(100): # add 1000ms at a time
                this_time_beg = this_time + i * stride_neg
                this_time_end = this_time_beg + gap_neg
                if this_time_end > ms_end:
                    break
                cmd = "ffmpeg -ss " + str(max(timestamp_bias, (this_time_beg + timestamp_bias))) + "ms -i \"" + file_path + "\" -crf 10 -an -vf \"fps=30, scale=-2:256\"" + \
                                " -to " + str(gap_neg) + "ms " + data_target + filename + "_s" + str(this_time_beg) + "_t" + str(gap_neg) + "_neg_3.mp4"
                anno_file.write(data_target + filename + "_s" + str(this_time_beg) + "_t" + str(gap_neg) + "_neg_3.mp4 3\n")
                anno_cmds.write(cmd+"\n")
                cmds.append(cmd)
                cn[3] += 1

generate_frameidx_from_raw(split='val')
print(labels)
print(cn)
