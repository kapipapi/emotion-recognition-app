import librosa
import numpy as np


def catch_audio_feature(x, sr):
    features = np.empty((0, 156))

    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    features = np.append(features, mfcc, axis=0)

    mel = librosa.feature.melspectrogram(y=x, sr=sr)
    features = np.append(features, mel, axis=0)

    s_abs = np.abs(librosa.stft(x))
    spec = librosa.feature.spectral_contrast(S=s_abs, sr=sr)
    features = np.append(features, spec, axis=0)

    y = librosa.effects.harmonic(x)
    tonne = librosa.feature.tonnetz(y=y, sr=sr)
    features = np.append(features, tonne, axis=0)

    return features
