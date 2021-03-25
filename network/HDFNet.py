# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : HDFNet.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import torch
import torch.nn as nn
from aCNN import computeOffset 
from torchvision.ops.deform_conv import DeformConv2d

from torchvision import models
from module.BaseBlocks import BasicConv2d
from utils.tensor_ops import cus_sample, upsample_add
from backbone.ResNet import Backbone_ResNet50_in1, Backbone_ResNet50_in3
from backbone.VGG import (
    Backbone_VGG19_in1,
    Backbone_VGG19_in3,
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)
from module.MyModules import (
    DDPM,
    DenseTransLayer,
)
import torch.nn.functional as F

class HDFNet_VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_VGG16, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.depth_trans16 = DenseTransLayer(512, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)
        self.depth_trans4 = DenseTransLayer(256, 64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_4 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data, in_depth):
        in_data_1 = self.encoder1(in_data)
        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d, in_data_1_d

        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)

        in_data_4_aux = self.depth_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_4_d, in_data_8_d, in_data_16_d

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8

        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4

        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        del out_data_4, in_data_4_aux, in_data_2

        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        del out_data_2, in_data_1

        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()



class HDFNet_VGG16_b(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_VGG16_b, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        
        net = models.vgg16_bn(pretrained=pretrained, progress=True)
        self.encoder1 = nn.Sequential(*list(net.children())[0][0:6])
        self.encoder2 = nn.Sequential(*list(net.children())[0][6:13])
        self.encoder4 = nn.Sequential(*list(net.children())[0][13:23])
        self.encoder8 = nn.Sequential(*list(net.children())[0][23:33])
        self.encoder16 = nn.Sequential(*list(net.children())[0][33:42])
        self.relu = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv5_1b = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)
        self.bn5_1b = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_2b = DeformConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2b = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_3b = nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False)
        self.bn5_3b = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        
        
        self.relu_d = nn.ReLU(inplace=True)
        self.pool4_d = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5_1b_d = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)
        self.bn5_1b_d = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_2b_d = DeformConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2b_d = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_3b_d = nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False)
        self.bn5_3b_d = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        

        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)
        
        self.relu_d = nn.ReLU(inplace=True)
        self.pool4_d = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.depth_trans16 = DenseTransLayer(512, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)
        self.depth_trans4 = DenseTransLayer(256, 64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_4 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data, in_depth):
        depth = in_depth.clone() 
        # 0.001
        in_data_1 = self.encoder1(in_data)
        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d, in_data_1_d

        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        
        
        h4 = self.pool4(in_data_8)
        depth = F.interpolate(depth, h4.size()[2:], mode='bilinear', align_corners=True)
        depth += torch.randn_like(depth)*0.000001
        offset = computeOffset(depth[:, 0], 3, 1)
        offset = F.pad(offset, (1,1,1,1), "constant", 0).contiguous().cuda(1)
        dg = self.bn5_3b(self.conv5_3b(self.bn5_2b(self.conv5_2b(self.bn5_1b(self.conv5_1b(h4)), offset))))
        inter = self.encoder16(in_data_8) + dg
        in_data_16 = self.relu(inter)
        
        h4_d = self.pool4_d(in_data_8_d)
        dg_d = self.bn5_3b_d(self.conv5_3b_d(self.bn5_2b_d(self.conv5_2b_d(self.bn5_1b_d(self.conv5_1b_d(h4_d)), offset))))
        
        inter_d = self.depth_encoder16(in_data_8_d) + dg_d
        in_data_16_d = self.relu_d(inter_d)
        

        in_data_4_aux = self.depth_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_4_d, in_data_8_d, in_data_16_d

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8

        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4

        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        del out_data_4, in_data_4_aux, in_data_2

        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        del out_data_2, in_data_1

        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()
    
    



class HDFNet_VGG16_b_learn(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_VGG16_b_learn, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        
        net = models.vgg16_bn(pretrained=pretrained, progress=True)
        self.encoder1 = nn.Sequential(*list(net.children())[0][0:6])
        self.encoder2 = nn.Sequential(*list(net.children())[0][6:13])
        self.encoder4 = nn.Sequential(*list(net.children())[0][13:23])
        self.encoder8 = nn.Sequential(*list(net.children())[0][23:33])
        self.encoder16 = nn.Sequential(*list(net.children())[0][33:42])

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.offset_depth4 = nn.Conv2d(256, 18, 3, padding = 1, bias = False)
        self.offset_rgb4 = nn.Conv2d(256, 18, 3, padding = 1, bias = False)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4_1b = nn.Conv2d(256, 32, kernel_size=1, stride=1, bias=False)
        self.bn4_1b = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.conv4_2b = DeformConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2b = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.conv4_3b = nn.Conv2d(32, 512, kernel_size=1, stride=1, bias=False)
        self.bn4_3b = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)

        self.relu4_d = nn.ReLU(inplace=True)
        self.pool3_d = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1b_d = nn.Conv2d(256, 32, kernel_size=1, stride=1, bias=False)
        self.bn4_1b_d = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.conv4_2b_d = DeformConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2b_d = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.conv4_3b_d = nn.Conv2d(32, 512, kernel_size=1, stride=1, bias=False)
        self.bn4_3b_d = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)



        self.offset_depth5 = nn.Conv2d(512, 18, 3, padding = 1, bias = False)
        self.offset_rgb5 = nn.Conv2d(512, 18, 3, padding = 1, bias = False)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5_1b = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)
        self.bn5_1b = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_2b = DeformConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2b = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_3b = nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False)
        self.bn5_3b = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        
        
        self.relu5_d = nn.ReLU(inplace=True)
        self.pool4_d = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5_1b_d = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)
        self.bn5_1b_d = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_2b_d = DeformConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2b_d = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv5_3b_d = nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False)
        self.bn5_3b_d = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        

        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)
        
        self.relu_d = nn.ReLU(inplace=True)
        self.pool4_d = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.depth_trans16 = DenseTransLayer(512, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)
        self.depth_trans4 = DenseTransLayer(256, 64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_4 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data, in_depth):
        depth = in_depth.clone() 
        # 0.001
        in_data_1 = self.encoder1(in_data)
        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d, in_data_1_d


        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        

        h4 = self.pool4(in_data_8)
        offset_rgb5 = self.offset_rgb5(h4)       
        dg = self.bn5_3b(self.conv5_3b(self.bn5_2b(self.conv5_2b(self.bn5_1b(self.conv5_1b(h4)), offset_rgb5))))
        inter = self.encoder16(in_data_8) + dg
        in_data_16 = self.relu5(inter)

      
        h4_d = self.pool4_d(in_data_8_d)
        offset_depth5 = self.offset_depth5(h4_d)
        dg_d = self.bn5_3b_d(self.conv5_3b_d(self.bn5_2b_d(self.conv5_2b_d(self.bn5_1b_d(self.conv5_1b_d(h4_d)), offset_depth5))))
        
        inter_d = self.depth_encoder16(in_data_8_d) + dg_d
        in_data_16_d = self.relu5_d(inter_d)
        

        in_data_4_aux = self.depth_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_4_d, in_data_8_d, in_data_16_d

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8

        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4

        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        del out_data_4, in_data_4_aux, in_data_2

        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        del out_data_2, in_data_1

        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()


    
class HDFNet_VGG19(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_VGG19, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG19_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG19_in1(pretrained=pretrained)

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.depth_trans16 = DenseTransLayer(512, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)
        self.depth_trans4 = DenseTransLayer(256, 64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_4 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data, in_depth):
        in_data_1 = self.encoder1(in_data)
        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d, in_data_1_d

        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)

        in_data_4_aux = self.depth_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_4_d, in_data_8_d, in_data_16_d

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8

        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4

        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        del out_data_4, in_data_4_aux, in_data_2

        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        del out_data_2, in_data_1

        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()


class HDFNet_Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(HDFNet_Res50, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder2, self.encoder4, self.encoder8, self.encoder16, self.encoder32 = Backbone_ResNet50_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
            self.depth_encoder32,
        ) = Backbone_ResNet50_in1(pretrained=pretrained)

        self.trans32 = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16 = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8 = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4 = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2 = nn.Conv2d(64, 64, kernel_size=1)

        self.depth_trans32 = DenseTransLayer(2048, 64)
        self.depth_trans16 = DenseTransLayer(1024, 64)
        self.depth_trans8 = DenseTransLayer(512, 64)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data, in_depth):
        in_data_2 = self.encoder2(in_data)
        del in_data
        in_data_2_d = self.depth_encoder2(in_depth)
        del in_depth
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)
        del in_data_2_d
        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        del in_data_4_d
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)
        in_data_32 = self.encoder32(in_data_16)
        in_data_32_d = self.depth_encoder32(in_data_16_d)

        in_data_8_aux = self.depth_trans8(in_data_8, in_data_8_d)
        del in_data_8_d
        in_data_16_aux = self.depth_trans16(in_data_16, in_data_16_d)
        del in_data_16_d
        in_data_32_aux = self.depth_trans32(in_data_32, in_data_32_d)
        del in_data_32_d

        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)
        in_data_32 = self.trans32(in_data_32)

        out_data_32 = self.upconv32(in_data_32)  # 1024
        del in_data_32
        out_data_16 = self.upsample_add(self.selfdc_32(out_data_32, in_data_32_aux), in_data_16)
        del out_data_32, in_data_32_aux, in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8
        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4
        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        del out_data_4, in_data_2
        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        del out_data_2
        out_data = self.classifier(out_data_1)
        del out_data_1
        return out_data.sigmoid()


if __name__ == "__main__":
    model_path = "../../HDFFile/output/HDFNet_Ablation/Model12/TestDCV3_SimpleCombineV1_ND_NL2/pth/state_final.pth"
    model = HDFNet_VGG16()
    model.load_state_dict(torch.load(model_path))
