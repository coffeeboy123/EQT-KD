from torch import nn
import torch.nn.functional as F
import torch.nn.quantized as nnq

class BNNeck(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck, self).__init__()
        self.return_f = return_f
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)

        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck3(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3, self).__init__()
        self.return_f = return_f
        # self.reduction = nn.Linear(input_dim, feat_dim)
        # self.bn = nn.BatchNorm1d(feat_dim)

        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(feat_dim)

        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        # before_neck = x.squeeze(dim=3).squeeze(dim=2)
        # after_neck = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)
        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck_qat(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck_qat, self).__init__()
        self.return_f = return_f
        # 4D 텐서 (N, C, 1, 1)을 받는 BatchNorm2d 사용
        self.bn = nn.BatchNorm2d(input_dim)
        # BN의 bias는 학습하지 않음
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        before_neck = x
        after_neck  = self.bn(before_neck)
        feat = after_neck.view(after_neck.size(0), -1)  # (N, C)

        is_quantized = isinstance(self.classifier, nnq.Linear) \
            or callable(getattr(self.classifier, "weight", None))

        if self.return_f:
            if is_quantized:
                score_post = self.classifier(feat)
                score_pre  = score_post
            else:
                wfq = getattr(self.classifier, "weight_fake_quant", None)
                ap  = getattr(self.classifier, "activation_post_process", None)
                weight = self.classifier.weight
                if wfq is not None:
                    weight = wfq(weight)
                bias = getattr(self.classifier, "bias", None)
                score_pre  = F.linear(feat, weight, bias)
                score_post = ap(score_pre) if ap is not None else score_pre

            pre_bn = before_neck.view(before_neck.size(0), -1)
            return feat, score_post, pre_bn, score_pre
        else:
            if is_quantized:
                out = self.classifier(feat)
            else:
                wfq = getattr(self.classifier, "weight_fake_quant", None)
                ap  = getattr(self.classifier, "activation_post_process", None)
                weight = self.classifier.weight
                if wfq is not None:
                    weight = wfq(weight)
                bias = getattr(self.classifier, "bias", None)
                out = F.linear(feat, weight, bias)
                if ap is not None:
                    out = ap(out)
            return out

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck3_qat(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3_qat, self).__init__()
        self.return_f = return_f
        # 1x1 Convolution으로 차원 축소: (N, input_dim, H, W) -> (N, feat_dim, H, W)
        self.reduction = nn.Conv2d(input_dim, feat_dim, 1, bias=False)
        # 4D 텐서를 받는 BatchNorm2d 사용 (BN2d는 (N, C, H, W)를 기대)
        self.bn = nn.BatchNorm2d(feat_dim)
        # BN의 bias는 학습하지 않음.
        self.bn.bias.requires_grad_(False)
        # 분류기: 최종 출력 피처는 feat_dim 차원 (flatten 후 적용)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        if x.size(2) != 1 or x.size(3) != 1:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        bn_out = self.bn(x)
        feat = bn_out.view(bn_out.size(0), -1)  # (N, feat_dim)

        # ----- 양자화된 Linear인지 먼저 체크 -----
        is_quantized = isinstance(self.classifier, nnq.Linear) \
            or callable(getattr(self.classifier, "weight", None))  # convert 후 weight가 메서드일 수 있음

        if self.return_f:
            if is_quantized:
                # convert 이후: pre/post 분리 불가 → 둘 다 동일
                score_post = self.classifier(feat)
                score_pre  = score_post
            else:
                wfq = getattr(self.classifier, "weight_fake_quant", None)
                ap  = getattr(self.classifier, "activation_post_process", None)
                weight = self.classifier.weight
                if wfq is not None:
                    weight = wfq(weight)
                bias = getattr(self.classifier, "bias", None)
                score_pre  = F.linear(feat, weight, bias)  # pre (activation 전)
                score_post = ap(score_pre) if ap is not None else score_pre

            pre_bn = x.view(x.size(0), -1)
            return feat, score_post, pre_bn, score_pre

        else:
            if is_quantized:
                out = self.classifier(feat)
            else:
                wfq = getattr(self.classifier, "weight_fake_quant", None)
                ap  = getattr(self.classifier, "activation_post_process", None)
                weight = self.classifier.weight
                if wfq is not None:
                    weight = wfq(weight)
                bias = getattr(self.classifier, "bias", None)
                out = F.linear(feat, weight, bias)
                if ap is not None:
                    out = ap(out)
            return out

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(self.weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x.squeeze(3).squeeze(2))
        if self.return_f:
            f = x
            x = self.classifier(x)
            return f, x, f
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            # For old pytorch, you may use kaiming_normal.
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)




class BNNeck_neck(nn.Module):
    def __init__(self, input_dim, return_f=False):
        super(BNNeck_neck, self).__init__()
        self.return_f = return_f
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(self.weights_init_kaiming)

    def forward(self, x):
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)


        return after_neck, before_neck

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck_classifier(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck_classifier, self).__init__()
        self.return_f = return_f
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):

        if self.return_f:
            score = self.classifier(x)
            return score
        else:
            x = self.classifier(x)
            return x
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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck3_neck(nn.Module):
    def __init__(self, input_dim, feat_dim, return_f=False):
        super(BNNeck3_neck, self).__init__()
        self.return_f = return_f
        # self.reduction = nn.Linear(input_dim, feat_dim)
        # self.bn = nn.BatchNorm1d(feat_dim)

        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False, groups=feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)

        self.bn.bias.requires_grad_(False)
        self.bn.apply(self.weights_init_kaiming)


    def forward(self, x):
        x = self.reduction(x)
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)

        return after_neck, before_neck

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck3_classifier(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3_classifier, self).__init__()
        self.return_f = return_f


        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):

        if self.return_f:
            score = self.classifier(x)
            return score
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck3_depthwise(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3_depthwise, self).__init__()
        self.return_f = return_f
        # self.reduction = nn.Linear(input_dim, feat_dim)
        # self.bn = nn.BatchNorm1d(feat_dim)

        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False, groups=feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)

        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        # before_neck = x.squeeze(dim=3).squeeze(dim=2)
        # after_neck = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)
        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

