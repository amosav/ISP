"""
This file will define the general utility functions you will need for you implementation throughout this ex.
We suggest you start with implementing and testing the functions in this file.

NOTE: each function has expected typing for it's input and output arguments. 
You can assume that no other input types will be given and that shapes etc. will be as described.
Please verify that you return correct shapes and types, failing to do so could impact the grade of the whole ex.

NOTE 2: We STRONGLY encourage you to write down these function by hand and not to use Copilot/ChatGPT/etc.
Implementaiton should be fairly simple and will contribute much to your understanding of the course material.

NOTE 3: You may use external packages for fft/stft, you are requested to implement the functions below to 
standardize shapes and types.
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


def create_single_sin_wave(frequency_in_hz, total_time_in_secs=3, sample_rate=16000):
    timesteps = np.arange(0, total_time_in_secs * sample_rate) / sample_rate
    sig = np.sin(2 * np.pi * frequency_in_hz * timesteps)
    return torch.Tensor(sig).float()


def load_wav(abs_path: tp.Union[str, Path]) -> tp.Tuple[torch.Tensor, int]:
    """
    This function loads an audio file (mp3, wav).
    If you are running on a computer with gpu, make sure the returned objects are mapped on cpu.

    abs_path: path to the audio file (str or Path)
    returns: (waveform, sample_rate)
        waveform: torch.Tensor (float) of shape [num_channels, T]
        sample_rate: int, the corresponding sample rate
    """
    waveform, sample_rate = ta.load(abs_path)
    return torch.tensor(waveform).cpu().unsqueeze(0), sample_rate

def do_stft(wav: torch.Tensor, n_fft: int = 1024) -> torch.Tensor:
    """
    This function performs STFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.stft.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    n_fft: int, denoting the number of used fft bins.

    returns: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    """
    if len(wav.shape) == 2:
        wav = wav.unsqueeze(0)
    all_stft = []
    for i in range(wav.shape[0]):
        cur_stft = torch.stft(input=wav[i],
                              n_fft=n_fft,
                              win_length=n_fft,
                              hop_length=n_fft//4,
                              return_complex=False)
        all_stft.append(cur_stft)
    final_stft = torch.cat(all_stft, dim=0)
    if final_stft.shape[0] > 1:
        final_stft = final_stft.unsqueeze(1)
    return final_stft



def do_istft(spec: torch.Tensor, n_fft: int=1024) -> torch.Tensor:
    """
    This function performs iSTFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.istft.

    spec: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    n_fft: int, denoting the number of used fft bins.

    returns: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.

    NOTE: you may need to use torch.view_as_complex.
    """
    istft_tensor = torch.istft(input=spec,
                                 n_fft=n_fft,
                                 win_length=n_fft,
                                 hop_length=n_fft//4,
                                 return_complex=True)
    return istft_tensor.view_as_complex()


def do_fft(wav: torch.Tensor) -> torch.Tensor:
    """
    This function performs fast fourier trasform (FFT) .

    hint: see scipy.fft.fft / torch.fft.rfft, you can convert the input tensor to numpy just make sure to cast it back to torch.

    wav: torch tensor of the shape (1, T).

    returns: corresponding FFT transformation considering ONLY POSITIVE frequencies, returned tensor should be of complex dtype.
    """
    fft_tensor = torch.fft.rfft(wav)
    return fft_tensor


def plot_spectrogram(wav: torch.Tensor, n_fft: int=1024, sr=16000) -> None:
    """
    This function plots the magnitude spectrogram corresponding to a given waveform.
    The Y axis should include frequencies in Hz and the x axis should include time in seconds.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.

    NOTE: for the batched case multiple plots should be generated (sequentially by order in batch)
    """
    plt.figure()
    sampled_wav = librosa.samples_to_time(wav, sr=sr)
    stft_tensor = do_stft(torch.tensor(sampled_wav), n_fft)
    magnitude = torch.sqrt(stft_tensor[..., 0]**2 + stft_tensor[..., 1]**2)
    if len(magnitude.shape) == 3:
        magnitude = magnitude.unsqueeze(0)
    num_plots = magnitude.shape[0]
    for i in range(magnitude.shape[0]):
        cur_magnitude = magnitude[i,0].cpu().numpy()
        cur_magnitude = librosa.amplitude_to_db(cur_magnitude)
        ax = plt.subplot(num_plots, 1, i+1)
        ax.imshow(cur_magnitude, aspect='auto', origin='lower')
    plt.show()

def plot_fft(wav: torch.Tensor) -> None:
    """
    This function plots the FFT transform to a given waveform.
    The X axis should include frequencies in Hz.

    NOTE: As abs(FFT) reflects around zero, please plot only the POSITIVE frequencies.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    """
    fft_tensor = do_fft(wav)
    magnitude = torch.abs(fft_tensor)
    if len(magnitude.shape) == 2:
        magnitude = magnitude.unsqueeze(0)
    num_plots = magnitude.shape[0]
    for i in range(magnitude.shape[0]):
        cur_magnitude = magnitude[i, 0].cpu().numpy()
        ax = plt.subplot(num_plots, 1, i+1)
        ax.plot(np.arange(cur_magnitude.shape[0]), librosa.db_to_amplitude(librosa.power_to_db(cur_magnitude)), c="r")
    plt.show()

