import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...localization import temporal_iop, temporal_iou
from ..builder import build_loss
from ..registry import NECKS
from .utils import post_processing
from mmcv.cnn import constant_init, normal_init, xavier_init

class Attention(nn.Module):
    def __init__(self, channels=512, dropout=0.4, type='channel'):
        super(Attention, self).__init__()
        self.channels = channels
        self.type = type
        self.q = nn.Conv2d(channels, channels, (3,3), padding=(1,1), bias=False)
        self.k = nn.Conv2d(channels, channels, (3,3), padding=(1,1), bias=False)
        self.v = nn.Conv2d(channels, channels, (3,3), padding=(1,1), bias=False)
        self.fc1 = nn.Conv2d(channels, channels, (1,1), bias=False)

        for l in [self.q, self.k, self.v, self.fc1]:
            nn.init.kaiming_normal_(l.weight)

    def forward_channel(self, x): 

        B, C, D, T = x.size()
        q = self.q(x).view(B, C, -1)
        k = self.k(x).view(B, C, -1)
        v = self.v(x).view(B, C, -1)
        
        q = q.permute(0,2,1).contiguous()

        attention = torch.matmul(q / self.channels**0.5, k)
        attention = F.softmax(attention, dim=-1)

        v = v.permute(0,2,1).contiguous()
        out = torch.matmul(attention, v)
        out = out.permute(0,2,1).contiguous().view(B, C, D, T)
        out += x
        out = self.fc1(out)
        return out

    def forward_spatial(self, x):
        B, C, D, T = x.size()
        q = self.q(x).view(B, C, -1) 
        k = self.k(x).view(B, C, -1)
        v = self.v(x).view(B, C, -1)

        k = k.permute(0,2,1).contiguous()

        attention = torch.matmul(q / self.channels**0.5, k)
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        out = out.view(B, C, D, T)
        out += x
        out = self.fc1(out)        
        return out

    def forward(self, x):
        if self.type=='channel':
            return self.forward_channel(x)
        return self.forward_spatial(x)

@NECKS.register_module()
class BMN_Neck(nn.Module):

    def __init__(self,
                 temporal_dim,
                 boundary_ratio,
                 num_samples,
                 num_samples_per_bin,
                 feat_dim,
                 soft_nms_alpha,
                 soft_nms_low_threshold,
                 soft_nms_high_threshold,
                 post_process_top_k,
                 feature_extraction_interval=16,
                 loss_cls=dict(type='BMNLoss'),
                 hidden_dim_1d=256,
                 hidden_dim_2d=128,
                 hidden_dim_3d=512):
        super().__init__()

        self.tscale = temporal_dim
        self.boundary_ratio = boundary_ratio
        self.num_samples = num_samples
        self.num_samples_per_bin = num_samples_per_bin
        self.feat_dim = feat_dim
        self.soft_nms_alpha = soft_nms_alpha
        self.soft_nms_low_threshold = soft_nms_low_threshold
        self.soft_nms_high_threshold = soft_nms_high_threshold
        self.post_process_top_k = post_process_top_k
        self.feature_extraction_interval = feature_extraction_interval
        self.loss_cls = build_loss(loss_cls)
        self.hidden_dim_1d = hidden_dim_1d
        self.hidden_dim_2d = hidden_dim_2d
        self.hidden_dim_3d = hidden_dim_3d

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(
                self.feat_dim,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1,
                groups=4), nn.ReLU(inplace=True),
            nn.Conv1d(
                self.hidden_dim_1d,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1,
                groups=4), nn.ReLU(inplace=True))

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim_1d,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1,
                groups=4), nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1), nn.Sigmoid())
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim_1d,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1,
                groups=4), nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1), nn.Sigmoid())

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim_1d,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1), nn.ReLU(inplace=True))
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(
                self.hidden_dim_1d,
                self.hidden_dim_3d,
                kernel_size=(self.num_samples, 1, 1)), nn.ReLU(inplace=True))
        
        self.x_2d_a1 = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim_3d,
                self.hidden_dim_3d,
                kernel_size=1),
            nn.ReLU(inplace=True))

        self.x_2d_a2 = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim_3d,
                self.hidden_dim_3d,
                kernel_size=1),
            nn.ReLU(inplace=True))

        self.atten_channel = Attention(self.hidden_dim_3d, 0.4, 'channel')
        self.atten_spatial = Attention(self.hidden_dim_3d, 0.4, 'spatial')
        
        """
        self.x_2d_m = nn.Sequential(
            nn.Conv2d(
                2*self.hidden_dim_3d, # 1024
                4*self.hidden_dim_3d,
                kernel_size=1),
            nn.ReLU(inplace=True))
        """
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(2*self.hidden_dim_3d, self.hidden_dim_3d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.hidden_dim_2d,
                self.hidden_dim_2d,
                kernel_size=3,
                padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(
                self.hidden_dim_2d,
                self.hidden_dim_2d,
                kernel_size=3,
                padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1), nn.Sigmoid())
        self.anchors_tmins, self.anchors_tmaxs = self._temporal_anchors(
            -0.5, 1.5)
        self.match_map = self._match_map()
        self.bm_mask = self._get_bm_mask()
        
        self.avg_pool1 = nn.AdaptiveAvgPool3d((None, 1,1))
        self.avg_pool2 = nn.AdaptiveAvgPool3d(1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def _match_map(self):
        """Generate match map."""
        temporal_gap = 1. / self.tscale
        match_map = []
        for idx in range(self.tscale):
            match_window = []
            tmin = temporal_gap * idx
            for jdx in range(1, self.tscale + 1):
                tmax = tmin + temporal_gap * jdx
                match_window.append([tmin, tmax])
            match_map.append(match_window)
        match_map = np.array(match_map)
        match_map = np.transpose(match_map, [1, 0, 2])
        match_map = np.reshape(match_map, [-1, 2])
        return match_map

    def _temporal_anchors(self, tmin_offset=0., tmax_offset=1.):
        """Generate temporal anchors.

        Args:
            tmin_offset (int): Offset for the minimum value of temporal anchor.
                Default: 0.
            tmax_offset (int): Offset for the maximun value of temporal anchor.
                Default: 1.

        Returns:
            tuple[Sequence[float]]: The minimum and maximum values of temporal
                anchors.
        """
        temporal_gap = 1. / self.tscale
        anchors_tmins = []
        anchors_tmaxs = []
        for i in range(self.tscale):
            anchors_tmins.append(temporal_gap * (i + tmin_offset))
            anchors_tmaxs.append(temporal_gap * (i + tmax_offset))

        return anchors_tmins, anchors_tmaxs

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.avg_pool1(x) # B, C, T, 1, 1
        x_t = self.avg_pool2(x).expand_as(x) # B, C, T, 1, 1
        x = torch.cat([x,x_t],1) # B, 2C, T, 1, 1
        x = x.squeeze(-1).squeeze(-1)

        base_feature = self.x_1d_b(x)
        # base_feature.shape [batch_size, self.hidden_dim_1d, self.tscale]
        start = self.x_1d_s(base_feature).squeeze(1)
        # start.shape [batch_size, self.tscale]
        end = self.x_1d_e(base_feature).squeeze(1)
        # end.shape [batch_size, self.tscale]
        confidence_map = self.x_1d_p(base_feature)
        # [batch_size, self.hidden_dim_1d, self.tscale]
        confidence_map = self._boundary_matching_layer(confidence_map)
        # [batch_size, self.hidden_dim_1d,, self.num_sampls, self.tscale, self.tscale] # noqa
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        # [batch_size, self.hidden_dim_3d, self.tscale, self.tscale]
        
        confidence_map_channel = self.x_2d_a1(confidence_map)
        confidence_map_channel = self.atten_channel(confidence_map_channel)
        # [batch_size, self.hidden_dim_3d, self.tscale, self.tscale]
        confidence_map_spatial = self.x_2d_a2(confidence_map)
        confidence_map_spatial = self.atten_spatial(confidence_map_spatial)

        confidence_map = torch.cat([confidence_map_spatial,confidence_map_channel], 1)
        # [batch_size, 2*self.hidden_dim_3d, self.tscale, self.tscale]
        
        confidence_map_feature = torch.max(confidence_map, axis=-1)[0]
        confidence_map_feature = confidence_map_feature.view(-1, 2*self.hidden_dim_3d, self.tscale, 1, 1)

        confidence_map = self.x_2d_p(confidence_map)
        # [batch_size, 2, self.tscale, self.tscale]

        return confidence_map, start, end, confidence_map_feature

    def _boundary_matching_layer(self, x):
        """Generate matching layer."""
        input_size = x.size()
        out = torch.matmul(x,
                           self.sample_mask).reshape(input_size[0],
                                                     input_size[1],
                                                     self.num_samples,
                                                     self.tscale, self.tscale)
        return out

    def forward_test(self, raw_feature, video_meta):
        """Define the computation performed at every call when testing."""
        if isinstance(raw_feature, tuple):
            raw_feature_slow, raw_feature_fast = raw_feature
        else:
            raw_feature_fast = raw_feature
        confidence_map, start, end, confidence_map_feature = self._forward(raw_feature_fast)
        start_scores = start[0].cpu().numpy()
        end_scores = end[0].cpu().numpy()
        cls_confidence = (confidence_map[0][1]).cpu().numpy()
        reg_confidence = (confidence_map[0][0]).cpu().numpy()

        max_start = max(start_scores)
        max_end = max(end_scores)

        # generate the set of start points and end points
        start_bins = np.zeros(len(start_scores))
        start_bins[0] = 1  # [1,0,0...,0,0]
        end_bins = np.zeros(len(end_scores))
        end_bins[-1] = 1  # [0,0,0...,0,1]
        for idx in range(1, self.tscale - 1):
            if start_scores[idx] > start_scores[
                    idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                start_bins[idx] = 1
            elif start_scores[idx] > (0.5 * max_start):
                start_bins[idx] = 1
            if end_scores[idx] > end_scores[
                    idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                end_bins[idx] = 1
            elif end_scores[idx] > (0.5 * max_end):
                end_bins[idx] = 1

        # iterate through all combinations of start_index and end_index
        new_proposals = []
        for idx in range(self.tscale):
            for jdx in range(self.tscale):
                start_index = jdx
                end_index = start_index + idx + 1
                if end_index < self.tscale and start_bins[
                        start_index] == 1 and end_bins[end_index] == 1:
                    tmin = start_index / self.tscale
                    tmax = end_index / self.tscale
                    tmin_score = start_scores[start_index]
                    tmax_score = end_scores[end_index]
                    cls_score = cls_confidence[idx, jdx]
                    reg_score = reg_confidence[idx, jdx]
                    score = tmin_score * tmax_score * cls_score * reg_score
                    new_proposals.append([
                        tmin, tmax, tmin_score, tmax_score, cls_score,
                        reg_score, score
                    ])
        new_proposals = np.stack(new_proposals)
        
        if not video_meta:
            fps = 30
            video_info = dict(duration_frame=32,duration_second=32/fps)
        else:
            video_info = dict(video_meta[0])
        proposal_list = post_processing(new_proposals, video_info,
                                        self.soft_nms_alpha,
                                        self.soft_nms_low_threshold,
                                        self.soft_nms_high_threshold,
                                        self.post_process_top_k,
                                        self.feature_extraction_interval)
        
        if not video_meta:
            return raw_feature, [proposal_list]
        
        video_name = video_info['video_name']
        output = [
            dict(
                video_name=video_name,
                proposal_list=proposal_list)
        ]      
        return raw_feature, output

    def forward_train(self, raw_feature, label_confidence, label_start,
                      label_end):
        """Define the computation performed at every call when training."""
        if isinstance(raw_feature, tuple):
            raw_feature_slow, raw_feature_fast = raw_feature
        else:
            raw_feature_fast = raw_feature
        confidence_map, start, end, confidence_map_feature = self._forward(raw_feature_fast)
        loss = self.loss_cls(confidence_map, start, end, label_confidence,
                             label_start, label_end,
                             self.bm_mask.to(raw_feature_fast.device))
        loss_dict = dict(loss_bmn=loss[0])
        return raw_feature,loss_dict, confidence_map_feature

    def generate_labels(self, gt_bbox):
        """Generate training labels."""
        match_score_confidence_list = []
        match_score_start_list = []
        match_score_end_list = []
        for every_gt_bbox in gt_bbox:
            gt_iou_map = []
            for start, end in every_gt_bbox:
                if isinstance(start, torch.Tensor):
                    start = start.data.cpu().numpy()
                if isinstance(end, torch.Tensor):
                    end = end.data.cpu().numpy()
                current_gt_iou_map = temporal_iou(self.match_map[:, 0],
                                                  self.match_map[:, 1], start,
                                                  end)
                current_gt_iou_map = np.reshape(current_gt_iou_map,
                                                [self.tscale, self.tscale])
                gt_iou_map.append(current_gt_iou_map)
            gt_iou_map = np.array(gt_iou_map).astype(np.float32)
            gt_iou_map = np.max(gt_iou_map, axis=0)

            gt_tmins = every_gt_bbox[:, 0].data.cpu().numpy()
            gt_tmaxs = every_gt_bbox[:, 1].data.cpu().numpy()

            gt_len_pad = 3 * (1. / self.tscale)

            gt_start_bboxs = np.stack(
                (gt_tmins - gt_len_pad / 2, gt_tmins + gt_len_pad / 2), axis=1)
            gt_end_bboxs = np.stack(
                (gt_tmaxs - gt_len_pad / 2, gt_tmaxs + gt_len_pad / 2), axis=1)

            match_score_start = []
            match_score_end = []

            for anchor_tmin, anchor_tmax in zip(self.anchors_tmins,
                                                self.anchors_tmaxs):
                match_score_start.append(
                    np.max(
                        temporal_iop(anchor_tmin, anchor_tmax,
                                     gt_start_bboxs[:, 0], gt_start_bboxs[:,
                                                                          1])))
                match_score_end.append(
                    np.max(
                        temporal_iop(anchor_tmin, anchor_tmax,
                                     gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
            match_score_confidence_list.append(gt_iou_map)
            match_score_start_list.append(match_score_start)
            match_score_end_list.append(match_score_end)
        match_score_confidence_list = torch.Tensor(match_score_confidence_list)
        match_score_start_list = torch.Tensor(match_score_start_list)
        match_score_end_list = torch.Tensor(match_score_end_list)
        return (match_score_confidence_list, match_score_start_list,
                match_score_end_list)

    def forward(self,
                raw_feature,
                gt_bbox=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            label_confidence, label_start, label_end = (
                self.generate_labels(gt_bbox))
            if isinstance(raw_feature, tuple):
                device = raw_feature[1].device
            else:
                device = raw_feature.device
            label_confidence = label_confidence.to(device)
            label_start = label_start.to(device)
            label_end = label_end.to(device)
            return self.forward_train(raw_feature, label_confidence,
                                      label_start, label_end)

        return self.forward_test(raw_feature, video_meta)

    @staticmethod
    def _get_interp1d_bin_mask(seg_tmin, seg_tmax, tscale, num_samples,
                               num_samples_per_bin):
        """Generate sample mask for a boundary-matching pair."""
        plen = float(seg_tmax - seg_tmin)
        plen_sample = plen / (num_samples * num_samples_per_bin - 1.0)
        total_samples = [
            seg_tmin + plen_sample * i
            for i in range(num_samples * num_samples_per_bin)
        ]
        p_mask = []
        for idx in range(num_samples):
            bin_samples = total_samples[idx * num_samples_per_bin:(idx + 1) *
                                        num_samples_per_bin]
            bin_vector = np.zeros(tscale)
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if 0 <= int(sample_down) <= (tscale - 1):
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if 0 <= int(sample_upper) <= (tscale - 1):
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_samples_per_bin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        """Generate sample mask for each point in Boundary-Matching Map."""
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.tscale):
                if start_index + duration_index < self.tscale:
                    p_tmin = start_index
                    p_tmax = start_index + duration_index
                    center_len = float(p_tmax - p_tmin) + 1
                    sample_tmin = p_tmin - (center_len * self.boundary_ratio)
                    sample_tmax = p_tmax + (center_len * self.boundary_ratio)
                    p_mask = self._get_interp1d_bin_mask(
                        sample_tmin, sample_tmax, self.tscale,
                        self.num_samples, self.num_samples_per_bin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_samples])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(
            torch.tensor(mask_mat).view(self.tscale, -1), requires_grad=False)

    def _get_bm_mask(self):
        """Generate Boundary-Matching Mask."""
        bm_mask = []
        for idx in range(self.tscale):
            mask_vector = [1] * (self.tscale - idx) + [0] * idx
            bm_mask.append(mask_vector)
        bm_mask = torch.tensor(bm_mask, dtype=torch.float)
        return bm_mask
