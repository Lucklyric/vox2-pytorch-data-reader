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
from vox2_data_reader import DataReader, fast_istft
from net import avNet
import scipy.io.wavfile as wavfile
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True


def weights_init_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def audio_loss(S1_pred, S1_true, S2_pred, S2_true, gamma=0.1):
    loss_A = (S1_true - S1_pred)**2 + (S2_true - S2_pred)**2
    loss_A = torch.mean(torch.sum(loss_A, [1, 3]), 1)
    loss_B = (S2_true - S1_pred)**2 + (S1_true - S2_pred)**2
    loss_B = torch.mean(torch.sum(loss_B, [1, 3]), 1)
    idx = (loss_A > loss_B).float()
    loss = torch.mean(idx * (loss_B) + (1 - idx) * loss_A)
    return loss
    # loss_B = torch.mean(loss_B)
    # if loss_A > loss_B:
    #     return loss_B    #- gamma*torch.mean((S1_true - S1_pred)**2 + (S2_true - S2_pred)**2)
    # else:
    #     return loss_A    #- gamma*torch.mean((S2_true - S1_pred)**2 + (S1_true - S2_pred)**2)


if __name__ == "__main__":
    CSV_META = "/mnt/hdd1/alvinsun/AV/vox2/vox2_meta.csv"    # "test with modified small scale meta csv"
    VIDEO_PREFIX = "/mnt/hdd1/alvinsun/AV/vox2/vox2_dev_mp4/dev/mp4/"    # "change to your own local path"
    AUDIO_PREFIX = "/mnt/hdd1/alvinsun/AV/vox2/vox2_aac/dev/aac/"    # "change to your own local path"
    AUDIO_PATH = "/mnt/hdd1/alvinsun/AV/speech_separation/data/audio/norm_audio_train/"    # "change to your own local path"
    BATCH_SIZE = 6
    SEED = 2019
    
    if not os.path.exists('log'):
        os.makedirs('log')
    writer = SummaryWriter('log/t1')

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
        random=True,
        engine="librosa",
        mode="train",
    )

    db_test = DataReader(
        csv_meta=CSV_META,
        audio_prefix=AUDIO_PREFIX,
        random=False,
        engine="librosa",
        mode="test",
    )

    device_0 = torch.device('cuda:0')

    db_loader_train = DataLoader(db_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)

    # db_loader_test = DataLoader(db_test, batch_size=3, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)
    evl_sample = db_test[0]

    evl_audio_mix = torch.tensor([evl_sample['audio_mix']]).float().to(device_0)
    evl_audio_s1 = torch.tensor([evl_sample['audio_s1']]).float().to(device_0)
    evl_audio_s2 = torch.tensor([evl_sample['audio_s2']]).float().to(device_0)
    evl_phase_mix = evl_sample['phase_mix']
    wav_mix = db_test.audio.spec2wav(np.squeeze(evl_audio_mix.cpu().data.numpy()), evl_phase_mix[0])
    wavfile.write("tmix.wav", 16000, wav_mix)

    plt.imsave('audio_s1_mag.png', evl_audio_s1[0, 0, :, :].cpu())
    plt.imsave('audio_s2_mag.png', evl_audio_s2[0, 0, :, :].cpu())
    plt.imsave('audio_mix_mag.png', evl_audio_mix[0, 0, :, :].cpu())
    plt.imsave('audio_s1_mag_mr.png', (evl_audio_s1 / (evl_audio_mix + 1e-8))[0, 0, :, :].cpu(), vmin=0, vmax=1, cmap='gray')
    plt.imsave('audio_s2_mag_mr.png', (evl_audio_s2 / (evl_audio_mix + 1e-8))[0, 0, :, :].cpu(), vmin=0, vmax=1, cmap='gray')

    # Define AV model

    model = avNet()
    print("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # model.apply(weights_init_kaiming)
    weights_init_kaiming(model)

    # criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    start_epoch = 0
    # model = nn.DataParallel(model)
    model.to(device_0)

    # Load model
    if os.path.exists('./ckpt'):
        print("Load ckpt file")
        ckpt = torch.load('./ckpt')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
    for pg in optimizer.param_groups:
        pg['lr'] = 0.00005

    # Train loops
    model.train()
    for epoch in range(start_epoch, 3000):
        # running_loss = 0.0
        for i_batch, sample_batched in enumerate(db_loader_train):
            audio_s1 = sample_batched['audio_s1'].to(device_0).float()
            audio_s2 = sample_batched['audio_s2'].to(device_0).float()
            audio_mix = sample_batched['audio_mix'].to(device_0).float()
            plt.imsave('batch_audio_s1_mag.png', audio_s1[0, 0, :, :].cpu())
            plt.imsave('batch_audio_s2_mag.png', audio_s2[0, 0, :, :].cpu())

            out_s1, out_s2 = model(audio_mix)

            # loss = criterion(out_s1 * audio_mix, audio_s1) + criterion(out_s2 * audio_mix, audio_s2)
            # loss = audio_loss(out_s1, audio_s1/(audio_mix+1e-8), out_s2, audio_s2/(audio_mix+1e-8), 0.1)
            loss = audio_loss(out_s1 * audio_mix, audio_s1, out_s2 * audio_mix, audio_s2, 0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            print('[%d, %d] loss: %.5f' % (epoch + 1, i_batch + 1, loss.item()))
            # running_loss = 0.0
            # awefawef
            if i_batch % 10 == 0:
                writer.add_scalar('loss',loss.item(),i_batch+epoch*(len(db_loader_train)//BATCH_SIZE))
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, "./ckpt")
                model.eval()
                evl_out_s1, evl_out_s2 = model(evl_audio_mix)
                # plt.imsave('evl_audio_s1_mag_rm.png', ((evl_out_s1)[0, 0, :, :]).cpu().data.numpy(), cmap='gray', vmin=0, vmax=1)
                # plt.imsave('evl_audio_s2_mag_rm.png', ((evl_out_s2)[0, 0, :, :]).cpu().data.numpy(), cmap='gray', vmin=0, vmax=1)
                plt.imsave('evl_audio_s1_mag_rm.png', ((evl_out_s1)[0, 0, :, :]).cpu().data.numpy(), cmap='gray', vmin=0, vmax=1)
                plt.imsave('evl_audio_s2_mag_rm.png', ((evl_out_s2)[0, 0, :, :]).cpu().data.numpy(), cmap='gray', vmin=0, vmax=1)
                evl_out_s1 = ((evl_audio_mix * evl_out_s1).cpu().data.numpy()).copy()
                evl_out_s2 = ((evl_audio_mix * evl_out_s2).cpu().data.numpy()).copy()
                plt.imsave('evl_audio_s1_mag.png', evl_out_s1[0, 0, :, :])
                plt.imsave('evl_audio_s2_mag.png', evl_out_s2[0, 0, :, :])
                wav_2 = db_test.audio.spec2wav(np.squeeze(evl_out_s2), evl_phase_mix[0])
                wav_1 = db_test.audio.spec2wav(np.squeeze(evl_out_s1), evl_phase_mix[0])
                wavfile.write("t1.wav", 16000, wav_1)
                wavfile.write("t2.wav", 16000, wav_2)

                print('Evl error: %f' %
                      np.mean(np.square(evl_out_s1 - evl_audio_s1.cpu().data.numpy()) + np.mean(np.square(evl_out_s2 - evl_audio_s2.cpu().data.numpy()))))
                model.train()
