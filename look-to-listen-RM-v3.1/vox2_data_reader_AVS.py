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


def power_law(data, power=0.6):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data >= 0] = 1
    mask[data < 0] = -1
    data = np.power(np.abs(data), power)
    data = data * mask
    return data


def stft(data, fft_size=512, step_size=160, padding=True):
    # short time fourier transform
    if padding == True:
        # for 16K sample rate data, 48192-192 = 48000
        pad = np.zeros(192,)
        data = np.concatenate((data, pad), axis=0)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)), np.hanning(fft_size - 112), np.zeros((56,))), axis=0)
    win_num = (len(data) - fft_size) // step_size
    out = np.ndarray((win_num, fft_size), dtype=data.dtype)
    for i in range(win_num):
        left = int(i * step_size)
        right = int(left + fft_size)
        out[i] = data[left:right] * window
    F = np.fft.rfft(out, axis=1)
    return F


def fast_stft(data, power=False):
    if power:
        data = power_law(data)
    return stft(data)


def istft(F, fft_size=512, step_size=160, padding=True):
    # inverse short time fourier transform
    print(F.shape)
    data = np.fft.irfft(F, axis=-1)
    print(data.shape)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)), np.hanning(fft_size - 112), np.zeros((56,))), axis=0)
    number_windows = F.shape[0]
    T = np.zeros((number_windows * step_size + fft_size))
    for i in range(number_windows):
        head = int(i * step_size)
        tail = int(head + fft_size)
        T[head:tail] = T[head:tail] + data[i, :] * window
    if padding == True:
        T = T[:48000]
    return T


def fast_istft(F, power=False, **kwargs):
    # directly transform the frequency domain data to time domain data
    # apply power law
    T = istft(F)
    if power:
        T = power_law(T, (1.0 / 0.6))
    return T


class DataReader(Dataset):

    def __init__(self, audio_path, random=False, engine="librosa", mode="train"):

        print("Init DataReader start")
        self.files = glob.glob("%s/*.wav" % (audio_path))
        self.files.sort()
        print(len(self.files))
        self.mode = mode
        if self.mode == "train":
            self.range = [0, 450]
        else:
            self.range = [450, 460]
        # print(self.range[0],self.range[1])

        self.files = self.files[self.range[0]:self.range[1]]
        self.length = len(self.files)
        self.random = random
        self.dur = int((1 / 25) * 16000)
        self.engine = engine

        print("Init DataReader finished")

    def __len__(self):
        # Return total numbe of speakers
        return self.length

    def get_single_data(self, idx):
        # np.random.seed(int(time.time() * 100000) % 100000)
        audio_path_wav = self.files[idx]

        raw_data, fs = librosa.load(audio_path_wav, sr=16000)
        Zxx = fast_stft(raw_data)
        Zxx = Zxx**0.3

        return raw_data, np.abs(Zxx)

    def __getitem__(self, idx):
        try:
            if self.random is True:
                ss = np.random.randint(0, self.length, 2)
            else:
                ss = [0, 1]
            raw_data_s1, mag_s1 = self.get_single_data(ss[0])
            raw_data_s2, mag_s2 = self.get_single_data(ss[1])
            mix_raw_data = 0.5 * raw_data_s1 + 0.5 * raw_data_s2
            Zxx = fast_stft(mix_raw_data)
            phase = np.angle(Zxx)
            Zxx = Zxx**0.3
            sample = {
                'audio_s1': np.asarray([mag_s1]),
                'audio_s2': np.asarray([mag_s2]),
                'audio_mix': np.asarray([np.abs(Zxx)]),
                'phase_mix': np.asarray([phase])
            }
            return sample

        except Exception as e:
            print(e)
            print("Error")
            return self.__getitem__(np.random_intel.randint(0, self.__len__()))

    def return_all_vidoes_pathes(self, id):
        """
        return all video pathes for a give human id
        """
        all_video_pathes = glob.glob(self.audio_prefix + id + "/*")
        all_video_pathes.sort()
        video_ids = [i.split("/")[-1] for i in all_video_pathes]
        return all_video_pathes, video_ids


if __name__ == "__main__":
    # CSV_META = "/mnt/hdd1/alvinsun/AV/vox2/vox2_meta.csv"
    AUDIO_PATH = "/mnt/hdd1/alvinsun/AV/speech_separation/data/audio/norm_audio_train/"    # "change to your own local path"
    BATCH_SIZE = 4
    SEED = 2019

    np.random.seed(SEED)

    def worker_init_fn(x):
        seed = SEED + x
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    db = DataReader(
        audio_path=AUDIO_PATH,
        random=True,
        engine="librosa",
        mode="train",
    )

    print("==================================")
    print("Test signle output:")
    test_single_input = db[0]

    print("Shape of s1 audio spectrogram: ", test_single_input["audio_s1"].shape)
    print("Shape of s2 audio spectrogram: ", test_single_input["audio_s2"].shape)
    print("Shape of mix audio spectrogram: ", test_single_input["audio_mix"].shape)
    # print("Shape of raw audio wave: ", test_single_input[""].shape)

    print("==================================")
    print("Test Batch dataloader output: with batch_size %d" % BATCH_SIZE)

    db_loader = DataLoader(db, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)

    for i_batch, sample_batched in enumerate(db_loader):
        print("num_batch %d" % i_batch)
        print("Shape of s1 audio spectrogram: ", sample_batched["audio_s1"].shape)
        print("Shape of mix audio spectrogram: ", sample_batched["audio_mix"].shape)
        print("")
        if i_batch == 1:
            break
