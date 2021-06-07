import json
import os
from multiprocessing import Process

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def cut_videos(cmds, i):
    for i in range(len(cmds)):
        cmd = cmds[i]
        print(i, cmd)
        os.system(cmd)
    return

cmds = [c.strip() for c in open("loveu_1s_cmds_6.txt").readlines()]

cpu_core = 25
mp_num = max(1, int(len(cmds)/cpu_core))
splits = list(chunks(cmds, mp_num))

ps = []
for i in range(len(splits)):
    p = Process(target=cut_videos, args=(splits[i], i))
    p.start()
    ps.append(p)
for p in ps:
    p.join()
