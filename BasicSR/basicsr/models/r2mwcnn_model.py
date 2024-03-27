import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.losses import build_loss
from collections import OrderedDict


@MODEL_REGISTRY.register()
class R2MWCNNModel(SRModel):
    def __init__(self, opt):
        super(R2MWCNNModel, self).__init__(opt)


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None
        
        if train_opt.get('channel_opt'):
            self.cri_channel = build_loss(train_opt['channel_opt']).to(self.device)
        else:
            self.cri_channel = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('region_opt'):
            self.cri_region = build_loss(train_opt['region_opt']).to(self.device)
        else:
            self.cri_region = None

        if train_opt.get('vit_opt'):
            self.cri_vit = build_loss(train_opt['vit_opt']).to(self.device)
        else:
            self.cri_vit = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_edge is None and self.cri_channel is None and self.cri_ssim is None and self.cri_region is None:
            raise ValueError('Both pixel and perceptual and edge and channel and ssim and region losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, self.feature = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            if l_pix is not None:
                l_total += l_pix
                loss_dict['l_pix'] = l_pix
        # perceptual loss(只有percet(vgg))
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # edge loss
        if self.cri_edge:
            l_edge = self.cri_edge(self.output, self.gt)
            if l_edge is not None:
                l_total += l_edge
                loss_dict['l_edge'] = l_edge
        # channel loss
        if self.cri_channel:
            l_cha = self.cri_channel(self.output, self.gt)
            l_total += l_cha
            loss_dict['l_cha'] = l_cha
        # ssim loss
        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            if l_ssim is not None:
                l_total += l_ssim
                loss_dict['l_ssim'] = l_ssim
        #region loss
        if self.cri_region:
            l_region = self.cri_region(self.output, self.gt)
            if l_region is not None:
                l_total += l_region
                loss_dict['l_region'] = l_region
        #vit loss
        if self.cri_vit:
            l_vit = self.cri_vit(self.feature, self.gt)
            if l_vit is not None:
                l_total += l_vit
                loss_dict['l_vit'] = l_vit

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, self.feature = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, self.feature = self.net_g(self.lq)
            self.net_g.train()



