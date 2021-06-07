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
        help='0: 1st part, 1: second part, 2:third part, 3:forth part') # 4 parts

    parser.add_argument(
        '--datapath',
        type=str,
        default="data/loveu_wide_val_2s_30fps/",
        help='loveu val data path')
    parser.add_argument(
        '--targetpath',
        type=str,
        default="data/valres/",
        help='target path')
    parser.add_argument(
        '--modelname',
        type=str,
        default="csn_4cls_2s_30fps",
        help='pth model name')
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help=('fps'))
    parser.add_argument(
        '--stride',
        type=int,
        default=8,
        help='stride')
    args = parser.parse_args()
    return args

def bmn_proposals(results,
                  num_videos,
                  max_avg_proposals=None,
                  num_res = 1,
                  thres=0.0,
                  ):

    bmn_res = []
    for result in results:
        #video_id = result['video_name']
        num_proposals = 0
        cur_video_proposals = []
        for proposal in result:
            t_start, t_end = proposal['segment']
            score = proposal['score']
            if score < thres: continue
            cur_video_proposals.append([t_start, t_end, score])
            num_proposals += 1
        if len(cur_video_proposals)==0: 
            bmn_res.append(np.array([-2021]))
            continue
        cur_video_proposals = np.array(cur_video_proposals)

        ratio = (max_avg_proposals * float(num_videos) / num_proposals)

        this_video_proposals = cur_video_proposals[:, :2]
        sort_idx = cur_video_proposals[:, 2].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :].astype(np.float32)

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)

        # For each video, compute temporal_iou scores among the retrieved proposals
        total_num_retrieved_proposals = 0
        # Sort proposals by score
        num_retrieved_proposals = np.minimum(
            int(this_video_proposals.shape[0] * ratio),
            this_video_proposals.shape[0])
        total_num_retrieved_proposals += num_retrieved_proposals
        this_video_proposals = this_video_proposals[:num_retrieved_proposals, :]
        
        #print(this_video_proposals)
        this_video_gebd_proposals = this_video_proposals.mean(axis=-1)
        num_res = min(num_res, len(this_video_gebd_proposals))
        this_video_gebd_top_proposal = this_video_gebd_proposals[:num_res]

        bmn_res.append(this_video_gebd_top_proposal)
    return bmn_res


def show_results(model, data, test, cn, args):
    frame_queue = sliceable_deque(maxlen=args.sample_length)
    result_queue = deque(maxlen=1)
    result_path = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_score.npy"
    # save results with different scores
    result_bmn_path = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal.npy"

    result_bmn_path_3 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.3.npy"
    result_bmn_path_4 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.4.npy"
    result_bmn_path_5 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.5.npy"
    result_bmn_path_6 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.6.npy"
    result_bmn_path_7 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.7.npy"
    result_bmn_path_8 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.8.npy"
    result_bmn_path_9 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.9.npy"
    result_bmn_path_95 = args.targetpath + test.split(".")[0] + "/" + args.modelname + "_proposal_0.95.npy"

    videoLoader = FrameCV(args.datapath + '/' + test, FPS=args.fps, transform="resize256", start=None, duration=None)
    frames = videoLoader.frames[:, :, :, ::-1]
    print(cn, test, frames.shape)

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
    bmn_results = []
    num_sub_videos = 0
    for i in range(int(frames.shape[0]/stride)):
        num_sub_videos += 1
        start_index = i * stride
        frame_queue = frames_padded[(start_index):(start_index + args.sample_length)][0::data['frame_interval']].copy()
        ret, scores, output_bmn = inference(model, data, args, frame_queue)
        score_list.append(scores)
        bmn_results.append(output_bmn)
    bmn_res = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.0,
                            )

    score_list = np.array(score_list)
    bmn_res = np.array(bmn_res)

    bmn_res3 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.3,
                            )
    bmn_res3 = np.array(bmn_res3)

    bmn_res4 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.4,
                            )
    bmn_res4 = np.array(bmn_res4)

    bmn_res5 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.5,
                            )
    bmn_res5 = np.array(bmn_res5)

    bmn_res6 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.6,
                            )
    bmn_res6 = np.array(bmn_res6)

    bmn_res7 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.7,
                            )
    bmn_res7 = np.array(bmn_res7)

    bmn_res8 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.8,
                            )
    bmn_res8 = np.array(bmn_res8)

    bmn_res9 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.9,
                            )
    bmn_res9 = np.array(bmn_res9)

    bmn_res95 = bmn_proposals(bmn_results,
                            num_videos=1,
                            max_avg_proposals=100,
                            num_res = 10,
                            thres=0.95,
                            )
    bmn_res95 = np.array(bmn_res95)

    score_list = np.array(score_list)
    bmn_res = np.array(bmn_res)
    #print(cn, test, frames.shape, score_list.shape)
    np.save(result_path, score_list)
    np.save(result_bmn_path, bmn_res)
    np.save(result_bmn_path_3, bmn_res3)
    np.save(result_bmn_path_4, bmn_res4)
    np.save(result_bmn_path_5, bmn_res5)
    np.save(result_bmn_path_6, bmn_res6)
    np.save(result_bmn_path_7, bmn_res7)
    np.save(result_bmn_path_8, bmn_res8)
    np.save(result_bmn_path_9, bmn_res9)
    np.save(result_bmn_path_95, bmn_res95)
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
        scores, output_bmn = model(return_loss=False, **cur_data)
    return True, scores[0], output_bmn[0]

def main():
    args = parse_args()

    args.device = torch.device(args.device)
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    data = dict(img_shape=None, modality='RGB', label=-1)

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['frame_interval']
            data['frame_interval'] = step['frame_interval']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
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

    num_split = 4
    if args.stride == 2:
        num_split = 16
    if args.stride == 4:
        num_split = 6
    if args.stride == 8:
        num_split = 8
    if args.stride == 16:
        num_split = 16

    quater_len = int(len(tests)/num_split)

    if args.half != num_split-1:
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

