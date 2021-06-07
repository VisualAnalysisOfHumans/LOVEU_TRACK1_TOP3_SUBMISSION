import os
import decord as de
from multiprocessing import Process

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def scale_videos(vids, i):
    with open("loveu_invalid_" + str(i) + ".txt", "w") as fw:
        for vid in vids:
            #vid_path = video_path + vid
            try:
                vr = de.VideoReader(vid)
            except:
                fw.write(vid + "\n") 
    return

lines = open("loveu_wide_val_1s_30fps_annotation.txt").readlines() #+ open("soccernet_0222_ann_valid_short_test.txt").readlines()

vids = [line.strip().split(" ")[0] for line in lines]

cpu_core = 25
mp_num = max(1, int(len(vids)/cpu_core))
splits = list(chunks(vids, mp_num))

ps = []
for i in range(len(splits)):
    p = Process(target=scale_videos, args=(splits[i], i))
    p.start()
    ps.append(p)
for p in ps:
    p.join()

