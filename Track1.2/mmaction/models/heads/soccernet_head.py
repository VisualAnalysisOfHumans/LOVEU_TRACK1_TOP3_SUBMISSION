import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ...core import top_k_accuracy

from ..registry import HEADS
from .base import BaseHead

"""
   add regression here
"""

class SE_Block(nn.Module): 
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in//reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _, _= x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x*y.expand_as(x)

@HEADS.register_module()
class SoccerNetHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='SoccerNetLoss', reg_loss_weight=1.0, cls_loss_weight=1.0),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.representation_size = 1024

        self.fc1 = nn.Linear(self.in_channels, self.representation_size)  # 2048 -> 1024
        self.fc2 = nn.Linear(self.representation_size, self.representation_size)
        self.fc_cls = nn.Linear(self.representation_size, self.num_classes) 
        self.fc_reg = nn.Linear(self.representation_size, 1) 

        # self.fc_cls = nn.Linear(self.in_channels, self.num_classes) 
        # self.fc_reg = nn.Linear(self.in_channels, 1) 

        self.is_se = True
        if self.is_se:
            self.se_block = SE_Block(2048, reduction=16)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc_reg, std=self.init_std)

        nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.kaiming_uniform_(self.fc2.weight, a=1)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        # [N, in_channels, 5, 7, 7]
        if self.is_se:
            x = self.se_block(x)

        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]

        #if self.dropout is not None:
        #    x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]

        x_after = x.view(x.shape[0], -1)
        x_after = F.relu(self.fc1(x_after))
        x_after = F.relu(self.fc2(x_after))
        # [N, in_channels]

        if self.dropout is not None:
            x_after = self.dropout(x_after)
        cls_score = self.fc_cls(x_after)
        reg_offset = self.fc_reg(x_after)
        # return cls_score, reg_offset

        output = {"result_cls": cls_score, "result_reg":reg_offset}
        return output

    def loss(self,
             cls_score, 
             gt_labels, 
             reg_offset, 
             gt_offset):
        """Calculate the loss given output ``cls_score/reg_offset``, target ``gt_labels/gt_offset``.
        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()

        if gt_labels.shape == torch.Size([]):
            gt_labels = gt_labels.unsqueeze(0)

        top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                    gt_labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc'] = torch.tensor(
            top_k_acc[0], device=cls_score.device)
        losses['top5_acc'] = torch.tensor(
            top_k_acc[1], device=cls_score.device)

        loss, loss_cls, loss_reg = self.loss_cls(cls_score, reg_offset, gt_labels, gt_offset)

        losses['loss'], losses['loss_cls'], losses['loss_reg'] = loss, loss_cls, loss_reg

        return losses


