import pickle
import numpy as np
import os
from scipy.ndimage import filters
from scipy import ndimage
from multiprocessing import Process
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

stride = 2
beg_th = 0.369
end_th = 9.481 

date = '0604'
results_with_date = f"eval_{date}/" #"eval_0507_1/"

# result path and model name to retrive C3D results
path_res = 'testres/'
cls_model1 = 'csn_2cls_1.5s_2st_24fps'
cls_model2 = 'csn_2cls_2s_2st_30fps'
cls_model3 = 'csn_4cls_1.5s_2st_24fps'
cls_model4 = 'csn_4cls_2s_2st_30fps'

# result path and model name to retrive temporal detection results
path_bmn_res1 = '../data/testres_csn_bmn_e12_0520/'
bmn_model1 = 'csn_bmn_4cls_2s_30fps_16bw_attn_e12'

if not os.path.exists(f'./{results_with_date}'):
    os.makedirs(f'./{results_with_date}', exist_ok=True)

def pair_bmn_scores(signal, scores, fps, bias, stride=8):
    sta_arr = []
    sco_arr = []
    cls_stride = 2
    m = stride // cls_stride
    scores = scores[:len(signal)*m]

    for i in range(len(scores)):
        if i%m == 0:
            start_time = float(i//m * stride) / fps - bias - float(0.5 * 32/30)
            time = signal[i//m]
            timestamp = start_time + time
            sta_arr.append(timestamp)
            sco_arr.append(scores[i])
    return sta_arr, sco_arr


def detect_bmn_boundary(signal, scores, score_th, window=3):
    sta_arr = []
    scores2 = scores.copy()
    for i in range(len(scores)):
        scores2[i] = float("%.4f" % scores[i])

    scores_tmp1 = [0] * window
    scores_tmp1.extend(signal)
    scores_tmp1.extend([0] * window)
    scores_tmp2 = [0] * window
    scores_tmp2.extend(scores2)
    scores_tmp2.extend([0] * window)

    for i in range(len(scores_tmp1) - 2 * window):
        f = signal[i]
        flag_continue = False
        for j in range(1, window+1):
            if not (scores_tmp2[i+window] >= scores_tmp2[i-j+window] and scores_tmp2[i+window] >= scores_tmp2[i+j+window]):
                flag_continue = True
                break
        if flag_continue:
            continue
        if scores_tmp2[i+window] >= score_th:
            sta_arr.append(f)

    return sta_arr

def retrive_bmn_boundary(signal, gap_min, num=1):
    sta_arr = []

    for timestamps in signal:
        #num = min(num, len(timestamps))
        timestamps = [timestamps] #timestamps[:num]

        tmp = []
        for timestamp in timestamps:
            if timestamp < 0: continue
            tmp.append(timestamp)
        if len(tmp) == 0: continue
        sta_arr.append(tmp)

    if len(sta_arr)==0: return []
    tmp_arr = sorted(np.concatenate(sta_arr))
    res_arr = []

    for timestamp in tmp_arr:
        if timestamp > beg_th and timestamp < end_th and ((len(res_arr) != 0 and timestamp - res_arr[-1] > gap_min) or len(res_arr) == 0):
            res_arr.append(timestamp)

    if len(res_arr)==0: return []
    gap = 0.95
    ext = 0.12
    sta_tmp = []
    last_stamp = res_arr[0]
    for timestamp in res_arr[1:]:
        if timestamp - last_stamp <= gap:
            sta_tmp.append(last_stamp - ext)
            last_stamp = timestamp + ext
        else:
            sta_tmp.append(last_stamp)
            last_stamp = timestamp
    sta_tmp.append(last_stamp)

    res_arr = sta_tmp.copy()

    return res_arr

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def detect_boundary(signal, fps, bias, score_th, gap_min, window=4, stride=2):
    sta_arr = []
    sco_arr = []
    signal2 = signal.copy()
    for i in range(signal.shape[0]):
        signal2[i] = float("%.4f" % signal[i])

    signal_tmp1 = [0] * window
    signal_tmp1.extend(signal)
    signal_tmp1.extend([0] * window)
    signal_tmp2 = [0] * window
    signal_tmp2.extend(signal2)
    signal_tmp2.extend([0] * window)

    for i in range(len(signal_tmp1) - 2 * window):
        f = float(i * stride) / fps - bias # s
        flag_continue = False
        for j in range(1, window+1):
            if not (signal_tmp2[i+window] >= signal_tmp2[i-j+window] and signal_tmp2[i+window] >= signal_tmp2[i+j+window]):
                flag_continue = True
                break
        if flag_continue:
            continue

        if signal_tmp2[i+window] >= score_th and f > beg_th and f < end_th and ((len(sta_arr) != 0 and f - sta_arr[-1] > gap_min) or len(sta_arr) == 0):
            sta_arr.append(f)
            sco_arr.append(signal_tmp1[i+window])
    
    if len(sta_arr)==0: return [], sco_arr
    gap = 0.95
    ext = 0.12
    sta_tmp = []
    last_stamp = sta_arr[0]
    for timestamp in sta_arr[1:]:
        if timestamp - last_stamp <= gap:
            sta_tmp.append(last_stamp - ext)
            last_stamp = timestamp + ext
        else:
            sta_tmp.append(last_stamp)
            last_stamp = timestamp
    sta_tmp.append(last_stamp)

    sta_arr = sta_tmp.copy()
    
    return sta_arr, sco_arr

def generate_part(val_vids_part, i):
    cn_0_3 = 0
    cn_4_7 = 0
    cn_8_11 = 0
    cn_12_100 = 0
    cn_len = 0
    gap_min = 0.8 #0.8
    
    val_pkl_bmn1_1 = results_with_date + "test_predicts_bmn1_1_part" + str(i) + ".pkl"
       
    val_pkl_cls1 = results_with_date + "test_predicts_cls1_part" + str(i) + ".pkl"
    val_pkl_cls2 = results_with_date + "test_predicts_cls2_part" + str(i) + ".pkl"
    val_pkl_cls3 = results_with_date + "test_predicts_cls3_part" + str(i) + ".pkl"
    val_pkl_cls4 = results_with_date + "test_predicts_cls4_part" + str(i) + ".pkl"

    val_dicts_1_1 = {}

    val_dicts_2_1 = {}

    val_dicts_3_1 = {}

    val_dicts_4_1 = {}

    val_dicts_bmn_reg_1_1 = {}

    cn_vid = [0, 0, 0, 0, 0]
    for vid_id in val_vids_part:

        model1_bmn_reg_1 = np.load(path_bmn_res1 + vid_id + f'/{bmn_model1}_proposal.npy', allow_pickle=True)
        model1 = np.load(path_res + vid_id + "/" + cls_model1 + ".npy") # csn_2cls_1.5s_2st_24fps
        model2 = np.load(path_res + vid_id + "/" + cls_model2 + ".npy") # csn_2cls_2s_2st_30fps
        model3 = np.load(path_res + vid_id + "/" + cls_model3 + ".npy") # csn_4cls_1.5s_2st_24fps
        model4 = np.load(path_res + vid_id + "/" + cls_model4 + ".npy") # csn_4cls_2s_2st_30fps

        # no bias in testing data
        this_bias = 0

        gap_dd = 0.7 # 0.8
        
        sta_arr, sco_arr = detect_boundary(model1[:, 0], 24, this_bias, 0.5, gap_dd) #0.93
        val_dicts_1_1[vid_id] = sta_arr.copy()

        sta_arr, sco_arr = detect_boundary(model2[:, 0], 30, this_bias, 0.5, gap_dd) #0.93
        val_dicts_2_1[vid_id] = sta_arr.copy()

        sta_arr, sco_arr = detect_boundary((1-model3[:, 3]), 24, this_bias, 0.5, gap_dd) #0.93
        val_dicts_3_1[vid_id] = sta_arr.copy()

        sta_arr, sco_arr = detect_boundary((1-model4[:, 3]), 30, this_bias, 0.5, gap_dd) #0.93
        val_dicts_4_1[vid_id] = sta_arr.copy()

        sta_arr, sco_arr = pair_bmn_scores(model1_bmn_reg_1, (model2[:, 0]), 30, this_bias, stride=8)
        sta_arr = detect_bmn_boundary(sta_arr, sco_arr, score_th=0.5, window=1) # 0.82
        sta_arr = retrive_bmn_boundary(sta_arr, gap_min=0.5, num=1) # 0.85
        val_dicts_bmn_reg_1_1[vid_id] = sta_arr.copy()
        
    output = open(val_pkl_cls1, 'wb')
    pickle.dump(val_dicts_1_1, output)
    output.close()

    output = open(val_pkl_cls2, 'wb')
    pickle.dump(val_dicts_2_1, output)
    output.close()

    output = open(val_pkl_cls3, 'wb')
    pickle.dump(val_dicts_3_1, output)
    output.close()

    output = open(val_pkl_cls4, 'wb')
    pickle.dump(val_dicts_4_1, output)
    output.close()

    output = open(val_pkl_bmn1_1, 'wb')
    pickle.dump(val_dicts_bmn_reg_1_1, output)
    output.close()

cpu_core = 25
mp_num = max(1, int(len(val_vids)/cpu_core) + 1)
splits = list(chunks(val_vids, mp_num))
print(len(splits))

ps = []
for i in range(len(splits)):
    p = Process(target=generate_part, args=(splits[i], i))
    p.start()
    ps.append(p)
for p in ps:
    p.join()

models = ['cls1', 'cls2', 'cls3', 'cls4', 'bmn1_1']

for model in models:
    cn = 0
    val_dicts_this_model = {}
    for part in range(25):
        model_path = results_with_date + "test_predicts_" + model + "_part" + str(part) + ".pkl"
        val_dicts_this_model.update(pickle.load(open(model_path, 'rb'), encoding='lartin1'))
    for vid_id in val_dicts_this_model:
        cn += len(val_dicts_this_model[vid_id])
    print(model, cn)
    output = open(results_with_date + "test_predicts_" + model + ".pkl", 'wb')
    pickle.dump(val_dicts_this_model, output)
    output.close()

