#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vox2_data_reader.py
# Copyright (c) 2019 Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy import random_intel
from pydub import AudioSegment
from skimage import io, transform, color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import imageio
import librosa
import numpy as np
import os
import pandas as pd
import time
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from vox2_data_reader import DataReader
from net import avNet
torch.backends.cudnn.benchmark = True


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


if __name__ == "__main__":
    CSV_META = "/mnt/hdd1/alvinsun/AV/vox2/vox2_meta_small.csv"    # "test with modified small scale meta csv"
    VIDEO_PREFIX = "/mnt/hdd1/alvinsun/AV/vox2/vox2_dev_mp4/dev/mp4/"    # "change to your own local path"
    AUDIO_PREFIX = "/mnt/hdd1/alvinsun/AV/vox2/vox2_aac/dev/aac/"    # "change to your own local path"
    BATCH_SIZE = 4
    SEED = 2019

    np.random.seed(SEED)

    # Configure Dataset


    def worker_init_fn(x):
        seed = SEED + x
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    db_train = DataReader(
        csv_meta=CSV_META,
        audio_prefix=AUDIO_PREFIX,
        video_prefix=VIDEO_PREFIX,
        random=True,
        engine="librosa",
        mode="train",
    )

    db_test = DataReader(
        csv_meta=CSV_META,
        audio_prefix=AUDIO_PREFIX,
        video_prefix=VIDEO_PREFIX,
        random=False,
        engine="librosa",
        mode="test",
    )

    device_0 = torch.device('cuda:0')

    db_loader_train = DataLoader(db_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)

    # db_loader_test = DataLoader(db_test, batch_size=3, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)
    evl_sample = db_test[0]

    evl_frames_s1 = torch.Tensor([evl_sample['frames_s1']]).float().to(device_0)
    evl_frames_s2 = torch.Tensor([evl_sample['frames_s2']]).float().to(device_0)
    evl_audio_mix = torch.Tensor([evl_sample['audio_mix']]).float().to(device_0)
    evl_audio_s1 = torch.Tensor([evl_sample['audio_s1']]).float().to(device_0)
    evl_audio_s2 = torch.Tensor([evl_sample['audio_s2']]).float().to(device_0)
    plt.imsave('audio_s1_real.png', evl_audio_s1[0, 0, :, :].cpu())
    plt.imsave('audio_s1_imag.png', evl_audio_s1[0, 1, :, :].cpu())
    plt.imsave('audio_s2_real.png', evl_audio_s2[0, 0, :, :].cpu())
    plt.imsave('audio_s2_imag.png', evl_audio_s2[0, 1, :, :].cpu())

    # Define AV model

    model = avNet()
    model.apply(weights_init_kaiming)
    model.to(device_0)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    start_epoch = 0
    # Load model
    if os.path.exists('./ckpt'):
        print("Load ckpt file")
        ckpt = torch.load('./ckpt')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']

    # Train loops
    model.train()
    for epoch in range(start_epoch, 3000):
        # running_loss = 0.0
        for i_batch, sample_batched in enumerate(db_loader_train):
            video_s1 = sample_batched['frames_s1'].to(device_0).float()
            video_s2 = sample_batched['frames_s2'].to(device_0).float()
            audio_s1 = sample_batched['audio_s1'].to(device_0).float()
            audio_s2 = sample_batched['audio_s2'].to(device_0).float()
            audio_mix = sample_batched['audio_mix'].to(device_0).float()

            optimizer.zero_grad()

            out_s1, out_s2 = model(video_s1, video_s2, audio_mix)

            loss = criterion(audio_mix*out_s1, audio_s1) + criterion(audio_mix*out_s2, audio_s2)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            print('[%d, %d] loss: %.5f' % (epoch + 1, i_batch + 1, loss.item()))
            # running_loss = 0.0

            if i_batch % 10 == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, "./ckpt")
                model.eval()
                evl_out_s1, evl_out_s2 = model(evl_frames_s1, evl_frames_s2, evl_audio_mix)
                plt.imsave('evl_audio_s1_real_rm.png', (evl_out_s1[0, 0, :, :]).cpu().data.numpy(),cmap='gray',vmin=0,vmax=1)
                plt.imsave('evl_audio_s1_imag_rm.png', (evl_out_s1[0, 1, :, :]).cpu().data.numpy(),cmap='gray',vmin=0,vmax=1)
                plt.imsave('evl_audio_s2_real_rm.png', (evl_out_s2[0, 0, :, :]).cpu().data.numpy(),cmap='gray',vmin=0,vmax=1)
                plt.imsave('evl_audio_s2_imag_rm.png', (evl_out_s2[0, 1, :, :]).cpu().data.numpy(),cmap='gray',vmin=0,vmax=1)
                evl_out_s1 = (evl_audio_mix*evl_out_s1).cpu().data.numpy()
                evl_out_s2 = (evl_audio_mix*evl_out_s2).cpu().data.numpy()
                plt.imsave('evl_audio_s1_real.png', evl_out_s1[0, 0, :, :])
                plt.imsave('evl_audio_s1_imag.png', evl_out_s1[0, 1, :, :])
                plt.imsave('evl_audio_s2_real.png', evl_out_s2[0, 0, :, :])
                plt.imsave('evl_audio_s2_imag.png', evl_out_s2[0, 1, :, :])
                print('Evl error: %f' %
                      np.mean(np.square(evl_out_s1 - evl_audio_s1.cpu().data.numpy()) + np.mean(np.square(evl_out_s2 - evl_audio_s2.cpu().data.numpy()))))
                model.train()
