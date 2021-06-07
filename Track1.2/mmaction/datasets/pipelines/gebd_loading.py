import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair
import random

from ..registry import PIPELINES

@PIPELINES.register_module()
class GEBD_Loading:
    """
    if clip长度为2s， fps是24，总共48帧
    if clip长度为1.5s， fps是24，总共36帧
    model input = 32帧
    """
    def __init__(self, window_size=32, len_clip_second = 2, sample_rate = 24, temporal_flip=True, thres=0.5): 
        self.len_clip_second = len_clip_second
        self.sample_rate = sample_rate
        self.frames = self.len_clip_second * self.sample_rate
        self.window_size = window_size
        self.temporal_flip = temporal_flip
        self.thres = thres

    def __call__(self, results):
        frame_inds = results['frame_inds']
        assert(len(frame_inds)==self.frames), "one clip should be {} frames but get {} frame inds".format(self.frames, len(frame_inds))
        if 'video_start' in results and 'video_end' in results:
            video_start = results['video_start']
            video_end = results['video_end']
            results['frame_inds'] = frame_inds[video_start : video_end]
            results['total_frames'] = len(results['frame_inds'])
            results['clip_len'] = self.window_size

            th = random.random()
            if self.temporal_flip and th > self.thres:
                results['frame_inds'] = np.flip(results['frame_inds'],axis=-1)

            assert (results['total_frames'] == self.window_size)
        return results
                    
