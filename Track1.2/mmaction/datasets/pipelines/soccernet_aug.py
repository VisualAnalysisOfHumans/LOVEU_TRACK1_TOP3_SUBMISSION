import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..registry import PIPELINES

@PIPELINES.register_module()
class SoccerNetAug:
    """
    clip长度为12s， fps是6，总共72帧
    标注时间在clip中间，窗口向前后滑动
    """
    def __init__(self, window_size=36, len_clip_second = 12, sample_rate = 6, multi_class=False): # action_spot在1.开始位置2.中间位置
        self.len_clip_second = len_clip_second
        self.sample_rate = sample_rate
        self.frames = self.len_clip_second * self.sample_rate
        self.window_size = window_size
        self.multi_class = multi_class
    def __call__(self, results):
        frame_inds = results['frame_inds']
        results['original_start_time'] = results['start_time']
        assert(len(frame_inds)==self.frames), "one clip should be {} frames but get {} frame inds".format(self.frames, len(frame_inds))

        if 'gt_offset' in results and 'start_time' in results and 'imgs' in results and 'clip_len' in results:
            gt_offset = results['gt_offset'] # frames
            start = random.randint(0, int(gt_offset))
            results['gt_offset'] = float(gt_offset - start)
            results['start_time'] = results['start_time'] + int(start/self.sample_rate)*1000
            results['frame_inds'] = frame_inds[start:start+self.window_size]
            results['total_frames'] = len(results['frame_inds'])
            results['clip_len'] = self.window_size
            results['imgs'] = results['imgs'][:,:,start:start+self.window_size,:,:]
            assert (results['total_frames'] == self.window_size)
            assert (len(results['imgs'][0]) == 3)
            assert (len(results['imgs'][0][0]) == 36)
        return results
