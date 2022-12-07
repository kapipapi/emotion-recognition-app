import librosa
import numpy as np
import torch


def get_mfccs(data: np.ndarray, sample_rate: int):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=10)
    return mfcc


def extract_features(y, sample_rate):
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    S = np.abs(librosa.stft(y))
    spec = librosa.feature.spectral_contrast(S=S, sr=sample_rate)
    z = librosa.effects.harmonic(y)
    tonne = librosa.feature.tonnetz(y=z, sr=sample_rate)
    feature_list = np.concatenate((mfcc, mel, spec, tonne), axis=0)
    feature_list = np.mean(feature_list, axis=1)
    feature_list = torch.from_numpy(feature_list).reshape((1, -1))
    # print("Audio, vector", feature_list.size(), type(feature_list))
    return feature_list
