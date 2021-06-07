import os.path as osp

import torch
import numpy as np
from .base import BaseDataset
from .registry import DATASETS
from ..core import average_recall_at_avg_proposals
from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)
import random
import copy
from collections import OrderedDict
from mmcv.utils import print_log

@DATASETS.register_module()
class GEBDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt
        video_path start_time end_time label
        some/path/000.mp4 3.2 4.3 1
        some/path/001.mp4 1.1 2.2 3
        some/path/002.mp4 2.3 2.7 4
        some/path/003.mp4 2.1 2.1 2
        some/path/004.mp4 3.3 3.3 3
        some/path/005.mp4 -1  -1  0


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        
        FPS = 30
        total_frames = 60 # 2s
        input_width = 32 # input frames
        VIDOE_DURATION = round(input_width/FPS, 3)
        boundary_width = 16
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for idx, line in enumerate(fin):
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                else:
                    filename, label = line_split
                    label = int(label)

                #if label == 3: continue # uncomment this line of code when finetune
                mid_frame = total_frames//2
                pad = input_width - boundary_width
                video_start = random.randint(max(0, mid_frame - boundary_width//2 - pad), min(mid_frame - boundary_width//2, total_frames-input_width))

                video_end = video_start + input_width
                start_time = round((mid_frame - boundary_width/2 - video_start)/FPS, 3)
                end_time = round((mid_frame + boundary_width/2 - video_start)/FPS, 3)
                
                segment = [start_time, end_time]
                # change the measurement from second to percentage
                current_start = max(min(1, segment[0]/VIDOE_DURATION), 0)
                current_end = max(min(1, segment[1]/VIDOE_DURATION), 0)
                gt_box = [[current_start, current_end]]
                gt_box = np.array(gt_box)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                
                video_name = filename.split('/')[-1].split('.')[0]
                duration_frame = input_width
                duration_second = VIDOE_DURATION
                video_infos.append(
                    dict(
                        video_start = video_start,
                        video_end = video_end,
                        filename=filename,
                        video_name = video_name,
                        label=onehot if self.multi_class else label,
                        gt_box=gt_box,
                        duration_frame = duration_frame,
                        duration_second = duration_second,
                        ))
        return video_infos
    
    def _import_ground_truth(self):
        """Read ground truth data from video_infos."""
        ground_truth = {}
        for video_info in self.video_infos:
            video_id = video_info['video_name']
            this_video_ground_truths = []
            # for ann in video_infos['annotations']:
            t_start, t_end = video_info['gt_box'][0]
            label = video_info['label']
            this_video_ground_truths.append([t_start, t_end, label])
            ground_truth[video_id] = np.array(this_video_ground_truths)
        return ground_truth

    @staticmethod
    def _import_proposals(results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name']
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options={
                     'top_k_accuracy' : dict(topk=(1, 5)),
                 },
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision', 'AR@AN'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]


        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels)
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results, gt_labels)
                eval_results['mean_average_precision'] = mAP
                log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue
        return eval_results


    def evaluate_bmn(
                    self,
                    results,
                    metrics='AR@AN',
                    metric_options={
                        'AR@AN':
                        dict(
                            max_avg_proposals=100,
                            temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))
                    },
                    logger=None,
                    **deprecated_kwargs):
        """Evaluation in feature dataset.

        Args:
            results (list[dict]): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'AR@AN'.
            metric_options (dict): Dict for metric options. Options are
                ``max_avg_proposals``, ``temporal_iou_thresholds`` for
                ``AR@AN``.
                default: ``{'AR@AN': dict(max_avg_proposals=100,
                temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))}``.
            logger (logging.Logger | None): Training logger. Defaults: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results for evaluation metrics.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['AR@AN'] = dict(metric_options['AR@AN'],
                                           **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        #allowed_metrics = ['AR@AN']
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision', 'AR@AN'
        ]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        ground_truth = self._import_ground_truth()
        proposal, num_proposals = self._import_proposals(results)

        for metric in metrics:
            if metric == 'AR@AN':
                temporal_iou_thresholds = metric_options.setdefault(
                    'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                            np.linspace(0.5, 0.95, 10))
                max_avg_proposals = metric_options.setdefault(
                    'AR@AN', {}).setdefault('max_avg_proposals', 100)
                if isinstance(temporal_iou_thresholds, list):
                    temporal_iou_thresholds = np.array(temporal_iou_thresholds)

                recall, _, _, auc = (
                    average_recall_at_avg_proposals(
                        ground_truth,
                        proposal,
                        num_proposals,
                        max_avg_proposals=max_avg_proposals,
                        temporal_iou_thresholds=temporal_iou_thresholds))
                eval_results['auc'] = auc
                eval_results['AR@1'] = np.mean(recall[:, 0])
                eval_results['AR@5'] = np.mean(recall[:, 4])
                eval_results['AR@10'] = np.mean(recall[:, 9])
                eval_results['AR@100'] = np.mean(recall[:, 99])

        return eval_results
