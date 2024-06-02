"""
This file will implement a digit classifier using rule-based dsp methods.
As all digit waveforms are given, we could take that under consideration, of our RULE-BASED system.

We reccomend you answer this after filling all functions in general_utilities.
"""
import numpy as np

from general_utilities import *

ROW_COL_DICT = get_row_col_dict()
IND_TO_DIGIT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
SPACE = -1
# --------------------------------------------------------------------------------------------------
#     Part A        Part A        Part A        Part A        Part A        Part A        Part A    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# In this part we will get familiarized with the basic utilities defined in general_utilities
# --------------------------------------------------------------------------------------------------


def self_check_fft_stft():
    """
    Q:
    1. create 1KHz and 3Khz sine waves, each of 3 seconds length with a sample rate of 16KHz.
    2. In a single plot (3 subplots), plot (i) FFT(sine(1Khz)) (ii) FFT(sine(3Khz)), 
       (iii) FFT(sine(1Khz) + sine(3Khz)), make sure X axis shows frequencies. 
       Use general_utilities.plot_fft
    3. concatate [sine(1Khz), sine(3Khz), sine(1Khz) + sine(3Khz)] along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    wave_1khz = create_single_sin_wave(1000, 3, 16000).unsqueeze(0)
    wave_3khz = create_single_sin_wave(3000, 3, 16000).unsqueeze(0)
    wave_1khz_3khz = wave_1khz + wave_3khz
    waves = torch.concatenate([wave_1khz.unsqueeze(0), wave_3khz.unsqueeze(0), wave_1khz_3khz.unsqueeze(0)], dim=0)
    plot_fft(waves)
    plot_spectrogram(waves)


def audio_check_fft_stft():
    """
    Q:
    1. load all phone_*.wav files in increasing order (0 to 11)
    2. In a single plot (2 subplots), plot (i) FFT(phone_1.wav) (ii) FFT(phone_2.wav). 
       Use general_utilities.plot_fft
    3. concatate all phone_*.wav files in increasing order (0 to 11) along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    phone_waves, sample_rate = load_phone_digits_waves()
    plot_fft(phone_waves[:2].unsqueeze(1))
    plot_spectrogram(phone_waves.unsqueeze(1), n_fft=1024, sr=sample_rate)



# --------------------------------------------------------------------------------------------------
#     Part B        Part B        Part B        Part B        Part B        Part B        Part B    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Digit Classifier
# --------------------------------------------------------------------------------------------------

def classify_single_digit(wav: torch.Tensor) -> int:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a given single digit waveform.
    Use ONLY functions from general_utilities file.

    Hint: try plotting the fft of all digits.
    
    wav: torch tensor of the shape (1, T).

    return: int, digit number
    """
    # this is not exactly if - else but it is base on cases.
    wav_fft = do_fft(wav)
    return get_number_from_fft(wav_fft)


def get_number_from_fft(wav_fft):
    peaks, _ = find_peaks(wav_fft.squeeze().numpy(), height=2)
    row_peak = min(peaks)
    col_peak = max(peaks)
    row_peaks = np.array(list(ROW_COL_DICT["row"].keys()))
    col_peaks = np.array(list(ROW_COL_DICT["col"].keys()))
    row_ind = np.argmin(np.abs(row_peaks - row_peak))
    col_ind = np.argmin(np.abs(col_peaks - col_peak))
    return IND_TO_DIGIT[row_ind * 3 + col_ind]


def classify_digit_stream(wav: torch.Tensor) -> tp.List[int]:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a waveform containing several digit stream.
    The input waveform will include at least a single digit in it.
    The input waveform will have digits waveforms concatenated on the temporal axis, with random zero
    padding in-between digits.
    You can assume that there will be at least 100ms of zero padding between digits
    The function should return a list of all integers pressed (in order).
    
    Use STFT from general_utilities file to answer this question.

    wav: torch tensor of the shape (1, T).

    return: List[int], all integers pressed (in order).
    """
    stft_tensor = do_stft(wav, 1024)
    magnitude = stft_tensor[..., 0]**2 + stft_tensor[..., 1]**2
    magnitude = magnitude.squeeze(0).cpu().numpy().T
    numbers = []
    for window in magnitude:
        if np.all(window == 0):
            numbers.append(SPACE)
            continue
        numbers.append(get_number_from_fft(torch.tensor(window)))