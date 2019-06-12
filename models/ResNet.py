from __future__ import absolute_import

import torch
import random
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from torch.autograd import Variable
from .resnet1 import ResNet
from .resnet_ibn import resnet50_ibn_a
from torch.nn import Parameter
import math
import IPython

__all__ = ['ResNet50', 'ResNet101', 'ResNet50M', 'MUB', 'AttPCB']

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight, std=0.001)
        init.constant_(m.bias, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
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
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        self.base = resnet50_ibn_a(last_stride=1)
        
        # resnet50 = torchvision.models.resnet50(pretrained=True)
        # resnet50.layer4[0].downsample[0].stride = (1,1)
        # resnet50.layer4[0].conv2.stride = (1,1)
        #self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048 # feature dimension
        self.num_classes = num_classes
        if num_classes is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5)
            )
            self.bottleneck.apply(weights_init_kaiming)
            if self.loss == 'sphereloss':
                self.classifier = AngleLinear(512, self.num_classes)
            else:
                self.classifier = nn.Linear(512, self.num_classes)
                self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        f_bottle = self.bottleneck(f)
        y = self.classifier(f_bottle)

        if self.loss == 'xent' or self.loss == 'sphereloss':
            return y
        elif self.loss == 'xent,htri':
            return y, f
        elif self.loss == 'cent':
            return y, f
        elif self.loss == 'ring':
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class AttPCB(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(AttPCB, self).__init__()
        self.loss = loss
        self.base = resnet50_ibn_a(last_stride=1)
        #self.base = ResNet(last_stride = 1)
        #self.base.load_param('/workspace/mnt/group/video/chenshuaijun/pytorch/models/resnet50-19c8e357.pth')
        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1,1)
        resnet50.layer4[0].conv2.stride = (1,1)
        #self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048 # feature dimension
        self.num_classes = num_classes


        # self.reduce_dim = nn.Conv2d(self.feat_dim, 512, kernel_size=1, stride=1)
        self.q_weight = nn.Conv2d(self.feat_dim, 512, kernel_size=1, stride=1)
        self.k_weight = nn.Conv2d(self.feat_dim, 512, kernel_size=1, stride=1)
        self.v_weight = nn.Conv2d(self.feat_dim, 512, kernel_size=1, stride=1)
        self.linear_out = nn.Conv2d(512, self.feat_dim, kernel_size=1, stride=1)
        self.part = 6
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))


        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512))

        # if num_classes is not None:
        #     self.bottleneck = nn.Sequential(
        #         nn.Linear(self.feat_dim, 512),
        #         nn.BatchNorm1d(512),
        #         nn.LeakyReLU(0.1),
        #         nn.Dropout(p=0.5)
        #     )
        #     self.bottleneck.apply(weights_init_kaiming)
        #     if self.loss == 'sphereloss':
        #         self.classifier = AngleLinear(512, self.num_classes)
        #     else:
        #         self.classifier = nn.Linear(512, self.num_classes)
        #         self.classifier.apply(weights_init_classifier)


    def attentionmodule(self, input_feat, group=8, fc_dim=64):
        N, C, P, _ = input_feat.size()
        output_feat = torch.ones_like(input_feat)
        for i in range(N):
            # parts_feat, [1,channel, part_num, 1]
            parts_feat = input_feat[i,...].unsqueeze(0)
            q_parts_feat = self.q_weight(parts_feat) # q_parts_feat, [1,512, 6, 1]
            q_parts_feat_reshape = q_parts_feat.view(group, fc_dim, P) # [8,64,6]
            q_parts_feat_permute = q_parts_feat_reshape.permute(0,2,1) # 8, 6, 64

            k_parts_feat = self.k_weight(parts_feat)
            k_parts_feat_reshape = k_parts_feat.view(group, fc_dim, P) # 8, 64, 6

            qk = torch.matmul(q_parts_feat_permute, k_parts_feat_reshape)
            qk_scale = 1.0 / math.sqrt(float(fc_dim)) * qk
            qk_softmax = torch.nn.Softmax(dim=2)(qk_scale)

            v_parts_feat = self.v_weight(parts_feat)
            v_parts_feat_reshape = v_parts_feat.view(group, fc_dim, P)
            v_parts_feat_permute = v_parts_feat_reshape.permute(0,2,1)


            output = torch.matmul(qk_softmax, v_parts_feat_permute) # [group, p, fc_dim]
            output = output.permute(0, 2, 1) # [group, fc_dim, p]
            output = output.contiguous().view(1, group*fc_dim, P, 1)
            output = self.linear_out(output)


            output_feat[i] = output.view(C, P, 1)

        return output_feat



    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)

        parts_f = self.avgpool(x)
        # reduced_parts_f = self.reduce_dim(parts_f)
        out_f = self.attentionmodule(parts_f, group=8, fc_dim=64)

        sum_f = parts_f + out_f

        if not self.training:
            sum_f = torch.squeeze(sum_f)
            return sum_f

        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(sum_f[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])

        if self.loss == 'xent' or self.loss == 'sphereloss':
            return y
        elif self.loss == 'xent,htri':
            return y, f
        elif self.loss == 'cent':
            return y, f
        elif self.loss == 'ring':
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class MUB(nn.Module):
    """
    ResNet50 + Multi-branch
    """
    def __init__(self, num_classes, loss={'xent'}, use_contraloss=False, **kwargs):
        super(MUB, self).__init__()
        self.use_contraloss = use_contraloss
        self.loss = loss
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.part = 6 # the numbers of parts
        self.scale = 30
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.resnet50.layer4[0].downsample[0].stride = (1,1)
        self.resnet50.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, num_classes, True, False, 256))
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        # global feature
        x_g = F.avg_pool2d(x, x.size()[2:])
        f_g = x_g.view(x.size(0), -1)
        #f_g = f_g/torch.norm(f_g, 2, 1, keepdim=True)
        #f_g = f_g * self.scale
        y_g = self.classifier(f_g)

        # cropped feature
        cp_h = int(x.size(2)/2)
        cp_w = int(x.size(3)/2)
        if self.training:
            start_h = random.randint(0, cp_h) 
            start_w = random.randint(0, cp_w)
        else:
            start_h = int(x.size(2)/4)
            start_w = int(x.size(3)/4)
        x_cp = x[:,:,start_h:start_h + cp_h, start_w:start_w+cp_w]
        x_cp = F.avg_pool2d(x_cp, x_cp.size()[2:])
        f_cp = x_cp.view(x_cp.size(0), -1)
        #f_cp = f_cp/torch.norm(f_cp, 2, 1, keepdim=True)
        #f_cp = f_cp * self.scale
        y_cp = self.classifier(f_cp)

        # part feature
        x_p = self.avgpool(x)
        #x_p=x_p/torch.norm(x_p,2,1,keepdim=True) * self.scale
        if not self.training:
            return f_g, f_cp, x_p
        x_p = self.dropout(x_p)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x_p[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y_p = []
        for i in range(self.part):
            y_p.append(predict[i])
        if self.loss == {'xent'}:
            if self.use_contraloss:
                return y_g, y_cp, y_p, f_g, f_cp
            else:
                return y_g, y_cp, y_p
        elif self.loss == {'xent', 'htri'}:
            return y_g, f_g, y_cp, f_cp, y_p, x_p
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))



def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)
