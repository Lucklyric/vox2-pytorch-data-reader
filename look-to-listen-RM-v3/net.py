# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:11:50 2019

@author: 60418 & Alvin(Xinyao) Sun
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt

# This net is for audio-stream #


class asNet(nn.Module):

    def __init__(self):

        super(asNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(1, 7), dilation=(1, 1), padding=(0, 3)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(7, 1), dilation=(1, 1), padding=(3, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(2, 1), padding=(4, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(4, 1), padding=(8, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(8, 1), padding=(16, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(16, 1), padding=(32, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(32, 1), padding=(64, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=2, padding=4),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=4, padding=8),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=8, padding=16),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=16, padding=32),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=32, padding=64),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=8, kernel_size=(1, 1), dilation=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)    # [batch, 8, 297, 257]

        out = torch.transpose(out, 1, 2)    # [batch , 297, 8, 257]
        out = torch.reshape(out, [-1, 298, 8 * 257])    # [bath, 297, 8*257]

        return out


# This net is video-stream #
class vsNet(nn.Module):

    def __init__(self):

        super(vsNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1000, out_channels=256, kernel_size=(7, 1), dilation=(1, 1), padding=(3, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False, dilation=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(4, 0), bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(8, 0), bias=False, dilation=(4, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(16, 0), bias=False, dilation=(8, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(32, 0), bias=False, dilation=(16, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.UpsamplingNearest2d((297, 1)),
        )

    def forward(self, x):
        out = self.net(x)    # [batch, 256, 297, 1]

        out = torch.transpose(out, 1, 2)    # [batch, 297, 256, 1]
        out = torch.reshape(out, [-1, 294, 256])    # [batch, 297, 256]

        return out


class fuseNet(nn.Module):

    def __init__(self):

        super(fuseNet, self).__init__()
        # self.biLSTMLayer = nn.LSTM(256 * 2 + 8 * 321, hidden_size=200, num_layers=1, bidirectional=True, batch_first=True)
        self.biLSTMLayer = nn.LSTM(8 * 257, hidden_size=400, num_layers=1, bidirectional=True, batch_first=True)
        self.fc3 = nn.Sequential(
            nn.Linear(800, 600), nn.ReLU(True), nn.Linear(600, 600), nn.ReLU(True), nn.Linear(600, 600), nn.ReLU(True), nn.Linear(600, 257 * 2 ), nn.ReLU()
            )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(400, 600), nn.ReLU(True), nn.Linear(600, 257 * 2 )
        #     )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(16*257, 600), nn.BatchNorm1d(298),nn.ReLU(True), nn.Linear(600, 257 * 2 ), nn.Sigmoid()
        #     )

        # self.fcMask1 = nn.Linear(600, 321 * 2 * 2)
        # self.fcMask2 = nn.Linear(600, 321 * 2)

    def forward(self, fuse_feature):
        print(fuse_feature.shape)
        plt.imsave("fuse_feature.png",fuse_feature[0].cpu().data.numpy());

        out, _ = self.biLSTMLayer(fuse_feature)    #[batch, 297, 800]
        
        # print(out.shape)
        plt.imsave("lstm_feature.png",out[0].cpu().data.numpy());
        out = self.fc3(out)
        # out = torch.sigmoid(self.fcMask1(out))
        out = torch.reshape(out, [-1, 298, 257, 1, 2])    # [batch , 297, 321 ,2]
        out = torch.transpose(out, 1, 2)
        out = torch.transpose(out, 1, 3)

        # out_s2 = torch.sigmoid(self.fcMask2(out))
        # out_s2 = torch.reshape(out_s2, [-1, 2, 297, 321])    # [batch , 297, 321 ,2]

        return out[:, :, :, :, 0], out[:, :, :, :, 1]


# fusion net #
class avNet(nn.Module):

    def __init__(self):

        super(avNet, self).__init__()
        self.asnet = asNet()
        self.fusenet = fuseNet()

    def forward(self, audio):
        audio_feature = self.asnet(audio)

        # self.fuse_feature = torch.cat([audio_feature, video_1_feature, video_2_feature], dim=2)
        self.fuse_feature = torch.cat([audio_feature], dim=2)

        print(self.fuse_feature.shape)

        out_s1, out_s2 = self.fusenet(self.fuse_feature)
        # print(out_s1.shape)
        return out_s1, out_s2


if __name__ == "__main__":
    device_0 = torch.device('cuda:1')
    device_1 = torch.device('cuda:1')
    device_2 = torch.device('cuda:2')

    # fake input
    test_video_1_input = torch.randn(3, 75, 1, 224, 224).to(device_0)
    test_video_2_input = torch.randn(3, 75, 1, 224, 224).to(device_0)
    test_audio_input = torch.randn(3, 1, 298, 257).to(device_0)

    # forward pass
    avnet = avNet().to(device_0)
    # avnet.to(device_0)
    # avnet.modified_vgg.to(device_0)
    # avnet.vsnet.to(device_1)
    # avnet.asnet.to(device_1)
    # avnet.fusenet.to(device_2)

    out_s1, out_s2 = avnet(test_video_1_input, test_video_2_input, test_audio_input)

    print(out_s1.shape)
