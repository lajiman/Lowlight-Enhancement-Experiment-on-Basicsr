import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models import _utils
import cv2
import numpy as np
from pytorch_msssim import ssim

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class SmoothL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, delta = 0.05):
        super(SmoothL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.delta = delta

    def forward(self, pred, target, **kwargs):
        diff = torch.abs(target - pred)
        loss = torch.where(diff<self.delta, 0.5 * diff ** 2, self.delta * diff - 0.5 * self.delta ** 2)
        return self.loss_weight * torch.mean(loss)


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction = 'mean'):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        ssim_ = 1 - ssim(pred, target)
        return ssim_ * self.loss_weight


@LOSS_REGISTRY.register()
class RegionLoss(nn.Module):
    def __init__(self, loss_weight=1.0, percent=0.3, delta=0.05):
        super(RegionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.percent = percent
        self.SmoothL1 = SmoothL1Loss(loss_weight=1.0, delta=delta)

    def forward(self, pred, target, **kwargs):
        # 计算灰度值
        gray1 = 0.39 * target[:, 0, ...] + 0.5 * target[:, 1, ...] + 0.11 * target[:, 2, ...]
        
        # 计算亮度百分比处的灰度值
        index = int(128 * 128 * self.percent - 1)
        gray_flat = gray1.view(-1, 128 * 128)
        yu = torch.topk(gray_flat, index + 1, dim=1)[0][:, index]
        
        # 创建掩码
        mask = (gray1 >= yu.view(-1, 1, 1)).float()
        mask = mask.unsqueeze(1)
        mask = torch.cat([mask, mask, mask], dim=1)
        
        # 计算损失
        low_fake_clean = mask * pred[:, :3, ...]
        high_fake_clean = (1 - mask) * pred[:, :3, ...]
        low_clean = mask * target[:, :, ...]
        high_clean = (1 - mask) * target[:, :, ...]
        Region_loss = self.SmoothL1(low_fake_clean, low_clean) * 4 + self.SmoothL1(high_fake_clean, high_clean)
        
        return Region_loss * self.loss_weight



@LOSS_REGISTRY.register()
class SobelEdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, delta=0.05):
        super(SobelEdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.SmoothL1 = SmoothL1Loss(loss_weight=1.0, delta=delta)

    def sobel_operator(self, image):
        device = image.device
        # 定义 Sobel 算子
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_x = torch.stack((sobel_x, sobel_x, sobel_x), dim=0).unsqueeze(0).to(device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        sobel_y = torch.stack((sobel_y, sobel_y, sobel_y), dim=0).unsqueeze(0).to(device)
        
        # 对图像应用 Sobel 算子
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        
        # 返回 Sobel 算子的结果
        return grad_x, grad_y

    def forward(self, pred, target, **kwargs):
        pred_edges = self.sobel_operator(pred)
        target_edges = self.sobel_operator(target)

        edge_loss1 = self.SmoothL1(pred_edges[0], target_edges[0])
        edge_loss2 = self.SmoothL1(pred_edges[1], target_edges[1])
        
        return (edge_loss1 + edge_loss2)*self.loss_weight


@LOSS_REGISTRY.register()
class ChannelLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ChannelLoss,self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        pred = torch.mean(pred, dim=[1, 2])
        target = torch.mean(target, dim=[1, 2])

        return torch.mean(torch.abs(pred - target)) * self.loss_weight


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
