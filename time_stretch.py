"""
In this file we will experiment with naively interpolating a signal on the time domain and on the frequency domain.

We reccomend you answer this file last.
"""
import torchaudio as ta
import soundfile as sf
import torch
import typing as tp
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np
from general_utilities import *
from torch.nn.functional import interpolate


def naive_time_stretch_temporal(wav: torch.Tensor, factor:float):
    """
    Q:
      write a function that uses a simple linear interpolation across the temporal dimension
      stretching/squeezing a given waveform by a given factor.
      Use imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    interpolated_wav = interpolate(wav, scale_factor=factor, mode='linear')
    return interpolated_wav

def naive_time_stretch_stft(wav: torch.Tensor, factor:float):
    """
    Q:
      write a function that converts a given waveform to stft, then uses a simple linear interpolation 
      across the temporal dimension stretching/squeezing by a given factor and converts the stretched signal 
      back using istft.
      Use general_utilities for STFT / iSTFT and imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """
    stft_wav = do_stft(wav)

if __name__ == '__main__':
    wav, sr = load_wav("audio_files/Basta_16k.wav")
    stretched_wav = naive_time_stretch_temporal(wav, 1.2)
    ta.save("audio_files/Basta_16k_stretched_1.2.wav", stretched_wav.squeeze(0), sr)
    stretched_wav = naive_time_stretch_temporal(wav, 0.8)
    ta.save("audio_files/Basta_16k_stretched_0.8.wav", stretched_wav.squeeze(0), sr)
    stretched_wav = naive_time_stretch_temporal(wav, 0.5)
    ta.save("audio_files/Basta_16k_stretched_stft_0.5.wav", stretched_wav.squeeze(0), sr)
