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
import matplotlib.pyplot as plt
from vox2_data_reader import DataReader
from net import avNet
torch.backends.cudnn.benchmark = True 
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

    db_loader_train = DataLoader(db_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)

    # Define AV model
    device_0 = torch.device('cuda:0')

    model = avNet()
    model.to(device_0)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Train loops
    for epoch in range(3000):
        # running_loss = 0.0
        for i_batch, sample_batched in enumerate(db_loader_train):
            video_s1 = sample_batched['frames_s1'].to(device_0).float()
            video_s2 = sample_batched['frames_s2'].to(device_0).float()
            audio_s1 = sample_batched['audio_s1'].to(device_0).float()
            audio_s2 = sample_batched['audio_s2'].to(device_0).float()
            audio_mix = sample_batched['audio_mix'].to(device_0).float()

            optimizer.zero_grad()

            out_s1, out_s2 = model(video_s1, video_s2, audio_mix)

            loss = criterion(out_s1, audio_s1 ) + criterion(out_s2, audio_s2)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            print('[%d, %d] loss: %.5f' % (epoch+1, i_batch+1, loss.item()))
            # running_loss = 0.0




