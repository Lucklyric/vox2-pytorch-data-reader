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


class DataReader(Dataset):

    def __init__(self, csv_meta, audio_prefix, video_prefix, random=False, engine="librosa", mode="train", fake_length=500):

        print("Init DataReader start")
        self.meta_data = pd.read_csv(csv_meta)
        self.meta_data = self.meta_data[self.meta_data["Set"].str.strip() == "dev"]
        self.ids = self.meta_data["VoxCeleb2 ID"].str.strip().values
        self.ids.sort()
        self.length = fake_length
        self.audio_prefix = audio_prefix
        self.video_prefix = video_prefix
        self.random = random
        self.dur = int((1 / 25) * 16000)
        self.engine = engine
        self.mode = mode

        print("Init DataReader finished")

    def __len__(self):
        # Return total numbe of speakers
        return self.length

    def get_single_data(self, idx):

        np.random.seed(int(time.time()*100000)%100000)
        speaker_id = self.ids[idx]

        # get all video ids of this speaker
        all_video_pathes, video_ids = self.return_all_vidoes_pathes(speaker_id)
        num_videos = len(all_video_pathes)

        pick_video_idx = 0
        if self.mode == 'train':
            # pick a random video of this speaker
            pick_video_idx = np.random.randint(1, num_videos - 1)
        else:
            pick_clip_idx = num_videos - 1

        # get all video clips of selected video
        all_audio_clips = glob.glob(all_video_pathes[pick_video_idx] + "/*.m4a")
        all_audio_clips.sort()
        num_audio_clips = len(all_audio_clips)

        pick_clip_idx = 0
        if self.random is True:
            # pick a random video clip
            pick_clip_idx = np.random.randint(0, num_audio_clips)

        # convert acc to wav
        audio_path_acc = all_audio_clips[pick_clip_idx]
        audio_path_wav = all_audio_clips[pick_clip_idx] + ".wav"
        if os.path.exists(audio_path_wav):
            pass
        else:
            acc = AudioSegment.from_file(audio_path_acc, "m4a")
            acc.export(audio_path_wav, format='wav')
        # print(audio_path_wav)

        if self.engine == "librosa":
            data, fs = librosa.load(audio_path_wav, sr=None)
            data = data / np.max(data)

        # locate video
        video_id = video_ids[pick_video_idx]

        video_path = "%s/%s/%s/%s.mp4" % (self.video_prefix, speaker_id, video_id, audio_path_wav.split('/')[-1].split('.')[0])
        # print(video_path)

        vid = imageio.get_reader(video_path, 'ffmpeg')

        num_frames = vid.get_length()

        # 3s segment
        len_seg = 3

        start_idx = 0

        if self.random is True:
            # pick random segments
            start_idx = np.random.randint(0, num_frames - 3 * 25 - 1)

        frames_seg = [np.expand_dims(color.rgb2gray(vid.get_data(x)), -1) for x in range(start_idx, start_idx + 3 * 25)]

        frames_seg = np.asarray(frames_seg)
        frames_seg = np.transpose(frames_seg, [0, 3, 1, 2])
        # frames_seg = np.transpose(frames_seg,1,3)

        dur = self.dur
        raw_data = data[start_idx * dur:(start_idx + 3 * 25 - 1) * dur]

        if len(raw_data) != 47360:
            print(data.shape)
            print(start_idx)
            print(vid.get_length())
            print(raw_data.shape)
            print(dur)

        if self.engine == "librosa":
            # raw_data = self.power_law(raw_data,0.3)
            Zxx = librosa.core.stft(raw_data.astype(float), hop_length=10 * 16, n_fft=40 * 16)
            Zxx = Zxx**0.3
            Zxx = np.transpose(Zxx, [1, 0])
            # # mag = np.abs(Zxx)
        return frames_seg, raw_data, np.abs(Zxx)

    def __getitem__(self, idx):
        try:
            frame_s1, raw_data_s1, mag_s1 = self.get_single_data(0)
            frame_s2, raw_data_s2, mag_s2 = self.get_single_data(1)
            # mix_raw_data = 0.5 * raw_data_s1 + 0.5 * raw_data_s2
            # mix_raw_data =  raw_data_s1 +  raw_data_s2
            # Zxx = librosa.core.stft(mix_raw_data.astype(float), hop_length=10 * 16, n_fft=40 * 16)
            # Zxx = Zxx**0.3
            Zxx = mag_s1 + mag_s2
            # Zxx = np.transpose(Zxx, [1, 0])
            sample = {
                'frames_s1': frame_s1,
                'frames_s2': frame_s2,
                'audio_s1': np.asarray([mag_s1]), 
                'audio_s2': np.asarray([mag_s2]),
                'audio_mix': np.asarray([np.abs(Zxx)])
            }
            return sample

        except Exception as e:
            print(e)
            print("Error")
            return self.__getitem__(np.random_intel.randint(0, self.__len__()))

    def power_law(self, data,power=0.6):
        mask = np.zeros(data.shape)
        mask[data>=0] = 1
        mask[data<0] = -1
        data = np.power(np.abs(data),power)
        data = data * mask
        return data

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
    CSV_META = "/mnt/hdd1/alvinsun/AV/vox2/vox2_meta_small.csv"    # "test with modified small scale meta csv"
    VIDEO_PREFIX = "/mnt/hdd1/alvinsun/AV/vox2/vox2_dev_mp4/dev/mp4/"    # "change to your own local path"
    AUDIO_PREFIX = "/mnt/hdd1/alvinsun/AV/vox2/vox2_aac/dev/aac/"    # "change to your own local path"
    BATCH_SIZE = 4
    SEED = 2019

    np.random.seed(SEED)

    def worker_init_fn(x):
        seed = SEED + x
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    db = DataReader(
        csv_meta=CSV_META,
        audio_prefix=AUDIO_PREFIX,
        video_prefix=VIDEO_PREFIX,
        random=True,
        engine="librosa",
        mode="train",
    )

    print("==================================")
    print("Test signle output:")
    test_single_input = db[0]

    print("Shape of s1 video frames: ", test_single_input["frames_s1"].shape)
    print("Shape of s2 video frames: ", test_single_input["frames_s2"].shape)
    print("Shape of s1 audio spectrogram: ", test_single_input["audio_s1"].shape)
    print("Shape of mix audio spectrogram: ", test_single_input["audio_mix"].shape)
    # print("Shape of raw audio wave: ", test_single_input[""].shape)

    print("==================================")
    print("Test Batch dataloader output: with batch_size %d" % BATCH_SIZE)

    db_loader = DataLoader(db, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn)

    for i_batch, sample_batched in enumerate(db_loader):
        print("num_batch %d" % i_batch)
        print("Shape of s1 video frames: ", sample_batched["frames_s1"].shape)
        print("Shape of s2 video frames: ", sample_batched["frames_s2"].shape)
        print("Shape of s1 audio spectrogram: ", sample_batched["audio_s1"].shape)
        print("Shape of mix audio spectrogram: ", sample_batched["audio_mix"].shape)
        print("")
        if i_batch == 3:
            break
