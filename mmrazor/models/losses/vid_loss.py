# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class VIDLoss(nn.Module):
    """"""

    def __init__(self,
                 num_student_channels: int,
                 num_mid_channels: int,
                 num_teacher_channels: int,
                 init_pred_var: float = 5.0,
                 eps: float = 1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels: int,
                    out_channels: int,
                    stride: int = 1) -> torch.Tensor:
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_student_channels, num_mid_channels),
            nn.ReLU(),
            conv1x1(num_mid_channels, num_mid_channels),
            nn.ReLU(),
            conv1x1(num_mid_channels, num_teacher_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) *
            torch.ones(num_teacher_channels))
        self.eps = eps

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        # pool for dimentsion match
        s_H, t_H = student_feature.shape[2], teacher_feature.shape[2]
        if s_H > t_H:
            student_feature = F.adaptive_avg_pool2d(student_feature,
                                                    (t_H, t_H))
        elif s_H < t_H:
            teacher_feature = F.adaptive_avg_pool2d(teacher_feature,
                                                    (s_H, s_H))
        else:
            pass

        pred_mean = self.regressor(student_feature)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
            (pred_mean - teacher_feature)**2 / pred_var + torch.log(pred_var))
        loss = torch.mean(neg_log_prob)
        return loss
