import torch
import numpy as np
from ..registry import RECOGNIZERS
from .base import BaseRecognizer
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


class SEModule(nn.Module):

    def __init__(self, channels, reduction=1/16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(channels, reduction)
        self.fc1 = nn.Conv3d(
            channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(
            self.bottleneck, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8):
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width,
                        int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

@RECOGNIZERS.register_module()
class GEBDRecognizer3D(BaseRecognizer):
    def __init__(self,
                 backbone,
                 cls_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 se_ratio=1/16,
                 **kwargs):
        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg, **kwargs)
        self.se_ratio = se_ratio
        if self.se_ratio is not None:
            self.se_module = SEModule(2048, self.se_ratio)

    def forward_train(self, imgs, labels, gt_box, **kwargs):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()
        
        x = self.extract_feat(imgs)
        if self.se_ratio is not None:
            x = self.se_module(x)

        if hasattr(self, 'neck'):
            x, loss_aux, confidence_map_feature = self.neck(x, gt_box, return_loss=True, **kwargs)

            losses.update(loss_aux)
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        
        losses.update(loss_cls)

        return losses

    def average_clip(self, cls_score, num_segs=1):
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=1)

        return cls_score

    def _do_test(self, imgs, **kwargs):
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            cls_scores = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if hasattr(self, 'neck'):
                    x, output_neck = self.neck(x, return_loss=False, **kwargs)
                cls_score = self.cls_head(x)
                cls_scores.append(cls_score)
                view_ptr += self.max_testing_views
            cls_score = torch.cat(cls_scores)
        else:
            x = self.extract_feat(imgs)
            if hasattr(self, 'neck'):
                x, output_neck = self.neck(x, return_loss=False, **kwargs)
            cls_score = self.cls_head(x)

        cls_score = self.average_clip(cls_score, num_segs)

        if hasattr(self, 'neck'):
            return cls_score.cpu().numpy(), output_neck
            
        return cls_score.cpu().numpy(), None

    def forward_test(self, imgs, **kwargs):
        return self._do_test(imgs, **kwargs)

    def forward_dummy(self, imgs, **kwargs):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if hasattr(self, 'neck'):
            x, _ = self.neck(x, **kwargs)

        outs = (self.cls_head(x), )
        return outs

    def forward_gradcam(self, imgs, **kwargs):
        return self._do_test(imgs, **kwargs)

    def forward(self, imgs, label=None, gt_box=None, return_loss=True, **kwargs):
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, label, gt_box, **kwargs)
        return self.forward_test(imgs, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        imgs = data_batch['imgs']
        label = data_batch['label']
        gt_box = data_batch['gt_box']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, gt_box, **aux_info)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        imgs = data_batch['imgs']
        label = data_batch['label']
        gt_box = data_batch['gt_box']

        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, gt_box, return_loss=True, **aux_info)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
