import librosa
import numpy as np
import torch


def get_mfccs(data: np.ndarray, sample_rate: int):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=10)
    return mfcc


def extract_features(data: np.ndarray, sample_rate: int):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    S = np.abs(librosa.stft(data))
    spec = librosa.feature.spectral_contrast(S=S, sr=sample_rate)
    z = librosa.effects.harmonic(data)
    tonne = librosa.feature.tonnetz(y=z, sr=sample_rate)

    feature_list = np.concatenate((mfcc, mel, spec, tonne), axis=0, dtype=float)

    print("Audio, vector", feature_list.shape, type(feature_list))

    return feature_list
