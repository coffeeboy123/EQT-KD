import copy
import torch
from torch import nn
from .osnet_qat import osnet_x0_5, OSBlock
from .attention import BatchDrop, PAM_Module, CAM_Module, SE_Module, Dual_Module, BatchFeatureErase_Top_osnet_x0_5
from .bnneck import BNNeck_qat, BNNeck3_qat
from torch.nn import functional as F

from torch.autograd import Variable

class LMBN_n_osnet_x0_5_student_scaling(nn.Module):
    def __init__(self, args):
        super(LMBN_n_osnet_x0_5_student_scaling, self).__init__()

        self.klw = nn.Linear(args.num_classes, 1, bias=False)  # (C) -> (1)
        self.klw.qconfig = None
        # 범위 하이퍼파라미터 (args에서 가져오고, 없으면 기본값)

        self.enable_klw = True

        self.quant = torch.quantization.QuantStub()  # 양자화 적용
        self.dequant = torch.quantization.DeQuantStub()  # 양자화 해제


        osnet = osnet_x0_5(pretrained=True)

        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )

        conv3 = osnet.conv3[1:]

        self.global_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.partial_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.channel_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.global_pooling = nn.MaxPool2d(kernel_size=(24, 8))
        self.partial_pooling = nn.AdaptiveAvgPool2d((12, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 2))
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        reduction = BNNeck3_qat(args.feats_student, args.num_classes,
                            args.feats_student, return_f=True)

        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)


        self.no_shared_1 = nn.Sequential(nn.Conv2d(
            args.feats_student, args.feats_student, 1, bias=False), nn.BatchNorm2d(args.feats_student), nn.ReLU(True))
        self.weights_init_kaiming(self.no_shared_1)

        self.no_shared_2 = nn.Sequential(nn.Conv2d(
            args.feats_student, args.feats_student, 1, bias=False), nn.BatchNorm2d(args.feats_student), nn.ReLU(True))
        self.weights_init_kaiming(self.no_shared_2)

        self.reduction_ch_0 = BNNeck_qat(
            args.feats_student, args.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck_qat(
            args.feats_student, args.num_classes, return_f=True)

        # if args.drop_block:
        #     print('Using batch random erasing block.')
        #     self.batch_drop_block = BatchRandomErasing()
        # print('Using batch drop block.')
        # self.batch_drop_block = BatchDrop(
        #     h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        self.batch_drop_block = BatchFeatureErase_Top_osnet_x0_5(args.feats_student, OSBlock)

        self.activation_map = args.activation_map

    def disable_klw_for_inference(self):
        self.enable_klw = False
        # (안전) 혹시라도 준비/변환 시 양자화 대상에서 제외
        if hasattr(self, "klw"):
            self.klw.qconfig = None

    def dequantize_if_needed(self, t):
        if isinstance(t, tuple):
            out = []
            for x in t:
                if torch.is_tensor(x) and x.is_quantized:
                    out.append(self.dequant(x))
                else:
                    out.append(x)
            return tuple(out)
        elif torch.is_tensor(t) and t.is_quantized:
            return self.dequant(t)
        else:
            return t

    def forward(self, x):
        # if self.batch_drop_block is not None:
        #     x = self.batch_drop_block(x)
        x = self.quant(x)
        x = self.backone(x)

        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)

        if self.activation_map:
            glo_ = glo

        if self.batch_drop_block is not None:
            glo_drop, glo = self.batch_drop_block(glo)

        if self.activation_map:

            _, _, h_par, _ = par.size()

            fmap_p0 = par[:, :, :h_par // 2, :]
            fmap_p1 = par[:, :, h_par // 2:, :]
            fmap_c0 = cha[:, :self.chs, :, :]
            fmap_c1 = cha[:, self.chs:, :, :]
            print('Generating activation maps...')

            return glo, glo_, fmap_c0, fmap_c1, fmap_p0, fmap_p1

        glo_drop = self.global_pooling(glo_drop)
        glo = self.average_pooling(glo)  # shape:(batchsize, 512,1,1)
        g_par = self.global_pooling(par)  # shape:(batchsize, 512,1,1)
        p_par = self.partial_pooling(par)  # shape:(batchsize, 512,2,1)
        cha = self.channel_pooling(cha)  # shape:(batchsize, 256,1,1)

        p_head = p_par[:, :, 0:2, :]
        p_upper = p_par[:, :, 2:7, :]
        p_lower = p_par[:, :, 7:12, :]

        p_head = F.adaptive_avg_pool2d(p_head, (1, 1))
        p_upper = F.adaptive_avg_pool2d(p_upper, (1, 1))
        p_lower = F.adaptive_avg_pool2d(p_lower, (1, 1))

        f_glo = self.reduction_0(glo)
        f_glo = self.dequantize_if_needed(f_glo)

        f_p0 = self.reduction_1(g_par)
        f_p0 = self.dequantize_if_needed(f_p0)

        f_p1 = self.reduction_2(p_head)
        f_p1 = self.dequantize_if_needed(f_p1)

        f_p2 = self.reduction_3(p_upper)
        f_p2 = self.dequantize_if_needed(f_p2)

        f_p3 = self.reduction_4(p_lower)
        f_p3 = self.dequantize_if_needed(f_p3)

        f_glo_drop = self.reduction_5(glo_drop)
        f_glo_drop = self.dequantize_if_needed(f_glo_drop)

        ################

        c0 = cha[:, :, :, 0:1]
        c1 = cha[:, :, :, 1:2]
        c0 = self.no_shared_1(c0)
        c1 = self.no_shared_2(c1)
        f_c0 = self.reduction_ch_0(c0)
        f_c0 = self.dequantize_if_needed(f_c0)

        f_c1 = self.reduction_ch_1(c1)
        f_c1 = self.dequantize_if_needed(f_c1)
        ################

        fea = [f_glo[2], f_glo_drop[2], f_p0[2]]

        s_post = torch.stack([f_glo[1], f_p0[1], f_p1[1], f_p2[1], f_p3[1], f_c0[1], f_c1[1]], dim=1)  # (B,H,C)
        s_pre = torch.stack([f_glo[3], f_p0[3], f_p1[3], f_p2[3], f_p3[3], f_c0[3], f_c1[3]], dim=1)  # (B,H,C)

        if self.enable_klw:
            diff = (s_post - s_pre).abs()  # (B,H,C)
            B, H, C = diff.shape
            w_raw = self.klw(diff.view(B * H, C))  # (B*H,1)
            w_learned = w_raw.view(B, H)  # (B,H)
            w_learned = 0.01*w_learned
        else:
            w_learned = None

        return [f_glo[1], f_p0[1], f_p1[1], f_p2[1], f_p3[1], f_c0[1], f_c1[1], f_glo_drop[1]], \
            fea, \
            torch.stack([f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_p3[0], f_c0[0], f_c1[0]], dim=2), \
            s_post, \
            s_pre, \
            w_learned

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)