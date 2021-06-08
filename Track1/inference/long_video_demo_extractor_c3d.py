import argparse
import random
import collections
import itertools
from collections import deque
from operator import itemgetter
import os
import cv2
import mmcv
import numpy as np
import torch
import json
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.core import OutputHook
from SoccerNet.utils import getListGames
from SoccerNet.DataLoader import Frame, FrameCV

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]

class sliceable_deque(collections.deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start,
                                               index.stop, index.step))
        return collections.deque.__getitem__(self, index)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 predict different labels in a long video demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--split',
        type=str,
        default="val",
        help='val/test')
    parser.add_argument(
        '--bias',
        type=int,
        default=1000,
        help='classify temporal bias')
    parser.add_argument(
        '--half',
        type=int,
        default=0,
        help='0: full, 1: first half, 2:second half, 3:, 4:') # 4 sections
    parser.add_argument(
        '--datapath',
        type=str,
        default="/home/notebook/data/personal/loveu/data/valset_24fps/",
        help='love val data path')
    parser.add_argument(
        '--targetpath', 
        type=str,
        default="/home/notebook/data/personal/loveu/data/valres/",
        help='soccernet target path')
    parser.add_argument(
        '--modelname',
        type=str,
        default="csn_4cls_1.5s_24fps",
        help='pth model name')
    parser.add_argument(
        '--fps',
        type=float,
        default=24.0,
        help=('fps'))
    parser.add_argument(
        '--stride',
        type=int,
        default=2,
        help='stride')
    args = parser.parse_args()
    return args

def show_results(model, data, test, cn, args):
    frame_queue = sliceable_deque(maxlen=args.sample_length)
    result_queue = deque(maxlen=1)
    result_path = args.targetpath + test.split(".")[0] + "/" + args.modelname + ".npy"
    videoLoader = FrameCV(args.datapath + test, FPS=args.fps, transform="resize256", start=None, duration=None)
    frames = videoLoader.frames[:, :, :, ::-1]

    duration = videoLoader.time_second
    stride = args.stride
    pad_length = int(args.sample_length/2)
    frames_head = np.zeros((pad_length, frames.shape[1], frames.shape[2], frames.shape[3]), frames.dtype)
    frames_tail = np.zeros((pad_length, frames.shape[1], frames.shape[2], frames.shape[3]), frames.dtype)
    for i in range(pad_length):
        frames_head[i] = frames[0].copy()
        frames_tail[i] = frames[-1].copy()
    frames_padded = np.concatenate((frames_head, frames, frames_tail), 0)
    score_list = []
    for i in range(int(frames.shape[0]/stride)):
        start_index = i * stride
        frame_queue = frames_padded[(start_index):(start_index + args.sample_length)][0::data['frame_interval']].copy()
        ret, scores = inference(model, data, args, frame_queue)
        score_list.append(scores)
    score_list = np.array(score_list)
    np.save(result_path, score_list)
    print(cn, result_path, "saved")

def inference(model, data, args, frame_queue):
    cur_windows = list(frame_queue)
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue[0].shape[:2]
    cur_data = data.copy()
    cur_data['imgs'] = cur_windows
    cur_data = args.test_pipeline(cur_data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [args.device])[0]
    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]
    return True, scores

def main():
    args = parse_args()

    args.device = torch.device(args.device)
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    data = dict(img_shape=None, modality='RGB', label=-1)

    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['frame_interval'] #step['num_clips']
            data['frame_interval'] = step['frame_interval']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0
    args.sample_length = sample_length
    args.test_pipeline = test_pipeline
   
    tests = []
    videos = os.listdir(args.datapath)
    for video in videos:
        video_path = video
        tests.append(video_path)

    quater_len = int(len(tests)/4) # 100/4

    if args.half != 3:
        tests = tests[(args.half*quater_len):((args.half+1)*quater_len)]
    else:
        tests = tests[(args.half*quater_len):]

    cn = 0
    for test in tests:
        cn += 1
        if not os.path.exists(args.targetpath + test.split(".")[0]):
            os.makedirs(args.targetpath + test.split(".")[0], exist_ok=True)
        show_results(model, data, test, cn, args)
if __name__ == '__main__':
    main()
