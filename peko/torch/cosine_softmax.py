import math

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(
        x.clamp(min=eps).pow(p),
        (x.size(-2), x.size(-1))
    ).pow(1./p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(
            self.p.data.tolist()[0]
        ) + ', ' + 'eps=' + str(self.eps) + ')'


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        with amp.autocast(enabled=False):
            input = input.float()

            # --------------------------- cos(theta) & phi(theta) ---------------------------
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            if self.ls_eps > 0:
                one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s

        return output


class ArcfaceModel(nn.Module):

    def __init__(self, n_classes=7770, model_name="resnet34", pooling="GeM",
                 margin=0.3, scale=30, fc_dim=512,
                 pretrained=None, loss_kwargs=None):
        super(ArcfaceModel, self).__init__()

        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained)

        if model_name.startswith("resne"):
            final_in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif model_name.startswith("dm_nfnet_") or model_name.startswith("tresnet_"):
            # dm_nfnet_f3 or dm_nfnet_f4
            final_in_features = self.backbone.head.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif model_name.startswith("swin_base_"):
            final_in_features = 1024
        else:
            raise NotImplementedError

        loss_kwargs = loss_kwargs or {
            "s": scale,
            "m": margin,
            "easy_margin": False,
            "ls_eps": 0.0,
        }

        self.pooling = GeM()

        # FC
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.final = ArcMarginProduct(fc_dim,
                                      n_classes,
                                      **loss_kwargs)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_features(x)

        with amp.autocast(enabled=False):
            logits = self.final(feature, label)

        return logits

    def extract_features(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x


class AngularModelChainHead(nn.Module):

    def __init__(self, n_classes=7770, model_name="resnet34", pooling="GeM",
                 n_chain_classes=88,
                 margin=0.3, scale=30, fc_dim=512,
                 pretrained=None, loss_kwargs=None):
        super(AngularModelChainHead, self).__init__()

        self.backbone = timm.create_model(
            model_name, num_classes=n_classes, pretrained=pretrained)

        if model_name.startswith("resne"):
            final_in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif model_name.startswith("dm_nfnet_"):
            # dm_nfnet_f3 or dm_nfnet_f4
            final_in_features = self.backbone.head.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise NotImplementedError

        loss_kwargs = loss_kwargs or {
            "s": scale,
            "m": margin,
            "easy_margin": False,
            "ls_eps": 0.0,
        }

        self.pooling = GeM()

        # chain head
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.dense1_1 = nn.Linear(final_in_features, 512)
        self.dense1_2 = nn.Linear(512, n_chain_classes)
        self.chain_softmax = nn.Softmax(dim=1)

        # FC
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(final_in_features,
                                      n_classes,
                                      **loss_kwargs)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = self.backbone(x)

        # chain head
        x1 = self.avgpool1(x).squeeze()
        x1 = self.dense1_1(x1)
        x1 = F.relu(x1)
        x1 = self.dense1_2(x1)
        chain_logits = self.chain_softmax(x1)

        # fc
        x = self.pooling(x).view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        logits = self.final(x, label)

        return logits, chain_logits

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        # fc
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OSME(nn.Module):
    def __init__(self, in_block, w, d, reduction=16):
        super(OSME, self).__init__()
        self.in_block = in_block
        self.reduction = reduction
        self.se1 = SEBlock(in_block, reduction=reduction)
        self.se2 = SEBlock(in_block, reduction=reduction)
        self.fc1 = nn.Linear(in_block * w * w, d)
        self.fc2 = nn.Linear(in_block * w * w, d)

    def forward(self, x):
        x1 = self.se1(x)
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.fc1(x1)

        x2 = self.se2(x)
        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.fc2(x2)

        return x1, x2


class AngularModelOSMEChainHead(nn.Module):

    def __init__(self, n_classes=7770, model_name="resnet34", pooling="GeM",
                 margin=0.3, scale=30, fc_dim=512,
                 pretrained=None, loss_kwargs=None):
        super(AngularModelOSMEChainHead, self).__init__()

        self.backbone = timm.create_model(
            model_name, num_classes=n_classes, pretrained=pretrained)

        final_in_features = self.backbone.fc.in_features

        loss_kwargs = loss_kwargs or {
            "s": scale,
            "m": margin,
            "easy_margin": False,
            "ls_eps": 0.0,
        }

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pooling = GeM()

        # chain head
        self.osme = OSME(512, 7, 1024)  # P=2, D=1024
        self.chain_fc = nn.Linear(2048, 88)  # P*D=2048
        self.chain_softmax = nn.Softmax(dim=1)

        # FC
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(final_in_features,
                                      n_classes,
                                      **loss_kwargs)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = self.backbone(x)

        # chain head
        x1_0, x1_1 = self.osme(x)
        x1 = torch.cat((x1_0, x1_1), dim=1)
        x1 = self.chain_fc(x1)
        x1 = self.chain_softmax(x1)

        # fc
        x = self.pooling(x).view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        logits = self.final(x, label)

        return logits, x1

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        # fc
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x


class AngularModelOSMEx2ChainHead(nn.Module):

    def __init__(self, n_classes=7770, model_name="resnet34", pooling="GeM",
                 margin=0.3, scale=30, fc_dim=512,
                 pretrained=None, loss_kwargs=None):
        super(AngularModelOSMEx2ChainHead, self).__init__()

        self.backbone_ = timm.create_model(
            model_name, num_classes=n_classes, pretrained=pretrained)

        final_in_features = self.backbone.fc.in_features

        loss_kwargs = loss_kwargs or {
            "s": scale,
            "m": margin,
            "easy_margin": False,
            "ls_eps": 0.0,
        }

        self.backbone = nn.Sequential(*list(self.backbone_.children())[:-3])
        self.backbone_last = list(self.backbone_.children())[-3]
        self.pooling = GeM()

        # chain head
        self.osme = OSME(512, 7, 1024)  # P=2, D=1024
        self.osme2 = OSME(256, 14, 1024)
        self.chain_fc = nn.Linear(4096, 88)  # P*D=2048
        self.chain_softmax = nn.Softmax(dim=1)

        # FC
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(final_in_features,
                                      n_classes,
                                      **loss_kwargs)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = self.backbone(x)

        x2_0, x2_1 = self.osme2(x)

        x = self.backbone_last(x)

        # chain head
        x1_0, x1_1 = self.osme(x)
        x1 = torch.cat((x1_0, x1_1, x2_0, x2_1), dim=1)
        x1 = self.chain_fc(x1)
        x1 = self.chain_softmax(x1)

        # fc
        x = self.pooling(x).view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        logits = self.final(x, label)

        return logits, x1

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.backbone_last(x)

        x = self.pooling(x).view(batch_size, -1)

        # fc
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x


class AngularModelOSMEx2ChainHotelHead(nn.Module):

    def __init__(self, n_classes=7770, model_name="resnet34", pooling="GeM",
                 margin=0.3, scale=30, fc_dim=512,
                 pretrained=None, loss_kwargs=None):
        super(AngularModelOSMEx2ChainHotelHead, self).__init__()
        self.backbone = timm.create_model(
            model_name, num_classes=n_classes, pretrained=pretrained)

        final_in_features = self.backbone.fc.in_features

        loss_kwargs = loss_kwargs or {
            "s": scale,
            "m": margin,
            "easy_margin": False,
            "ls_eps": 0.0,
        }

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pooling = GeM()

        # chain head
        self.osme = OSME(512, 7, 1024)  # P=2, D=1024
        self.chain_fc = nn.Linear(2048, 88)  # P*D=2048
        self.chain_softmax = nn.Softmax(dim=1)

        # hotel head
        self.osme2 = OSME(512, 7, 1024)  # P=2, D=1024
        self.hotel_fc = nn.Linear(2048, 7770)  # P*D=2048
        self.hotel_softmax = nn.Softmax(dim=1)

        # FC
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(final_in_features,
                                      n_classes,
                                      **loss_kwargs)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = self.backbone(x)

        # chain head
        x1_0, x1_1 = self.osme(x)
        x1 = torch.cat((x1_0, x1_1), dim=1)
        x1 = self.chain_fc(x1)
        x1 = self.chain_softmax(x1)

        # hotel head
        x2_0, x2_1 = self.osme2(x)
        x2 = torch.cat((x2_0, x2_1), dim=1)
        x2 = self.hotel_fc(x2)
        x2 = self.hotel_softmax(x2)

        # fc
        x = self.pooling(x).view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        logits = self.final(x, label)

        return logits, x1, x2

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        # fc
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x
