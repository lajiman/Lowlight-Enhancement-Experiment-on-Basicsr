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

from transformers import ViTModel


@LOSS_REGISTRY.register()
class VITLoss(nn.Module):
    def __init__(self, loss_weight = 1.0, model_name = 'facebook/vit-mae-base'):
        super(VITLoss, self).__init__()
        self.loss_weight = loss_weight
        self.vit_model = ViTModel.from_pretrained(model_name)

    def forward(self, feature_pred, target, **kwargs):
        with torch.no_grad():
            target = nn.functional.interpolate(target, (224,224), mode='bilinear', align_corners=True)
            feature_target = self.vit_model(target)
            feature_target = feature_target.last_hidden_state
            feature_target_reshape = feature_target.view(-1, feature_target.shape[-2], feature_target.shape[-1])

        feature_pred_reshape = feature_pred.view(-1, feature_pred.shape[-2], feature_pred.shape[-1])

        Up, Sp, Vp = torch.svd(feature_pred)
        Ut, St, Vt = torch.svd(feature_target)

        return(torch.abs(Sp.sum() - St.sum()) * self.loss_weight)

