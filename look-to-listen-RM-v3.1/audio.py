# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np


class Audio():

    def __init__(self, ref_level_db=20.0, hop_length=160, win_length=400, min_level_db=-100.0, sr=16000, n_fft=512):
        self.ref_level_db = ref_level_db
        self.hop_length = hop_length
        self.win_length = win_length
        self.min_level_db = min_level_db
        self.n_fft = 512
        self.sr = sr

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T    # to make [time, freq]
        return S, D

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + self.ref_level_db)
        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def istft(self, mag, phase):
        print(mag.shape, phase.shape)
        stft_matrix = mag * np.exp(1j * phase)
        return librosa.istft(stft_matrix, hop_length=self.hop_length, win_length=self.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.min_level_db
