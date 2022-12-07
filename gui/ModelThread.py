import time

import cv2
import numpy as np
import torch
import threading

from gui.CombinedSource import AVCapture
from gui.audio_preprocess import get_mfccs


class ModelThread:
    emotions = ["neutral", "calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']

    def __init__(self, capture: AVCapture, model: torch.nn.Module = None, device: torch.device = None):
        self.capture = capture

        self.started = False
        self.thread = None

        self.device = device
        self.model = model

        self.load_model()
        self.warm_model()

    def load_model(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.model is not None

        self.model.to(self.device)
        if self.is_cuda():
            self.model.cuda(self.device)

        self.model.eval()

    def is_cuda(self):
        return self.device == torch.device('cuda')

    def warm_model(self):
        print("[!] Model warmup")
        audio_rand = np.random.random([int(22050 * 3.6)]).astype('f')
        audio_rand = self.get_audio_tensor(audio_rand)

        video_rand = np.random.random([7, 256, 265, 3]).astype('f')
        video_rand = self.get_video_tensor(video_rand)

        self.model(audio_rand, video_rand)

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(
            target=self.update,
            args=()
        )
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (length_audio, audio_data), video_data = self.capture.read()

            if length_audio != self.capture.audio.nb_samples or \
                    len(video_data) != self.capture.video.nb_samples:
                time.sleep(3)
                continue

            audio = self.get_audio_tensor(audio_data)
            if audio.shape != torch.Size([1, 10, 156]):
                print("wrongaudio.shape:", audio.shape, "want [1, 10, 156]")
                continue

            video = self.get_video_tensor(video_data)
            if video.shape != torch.Size([1, 3, 7, 64, 64]):
                print("wrong video.shape:", video.shape, "want [1, 3, 7, 64, 64]")
                continue

            with torch.no_grad():
                output = self.model(audio, video)

            output = np.argmax(torch.nn.functional.softmax(output, dim=1).tolist())

            print("model output:", self.emotions[output])

    def get_video_tensor(self, video) -> torch.Tensor:
        video = np.array([cv2.resize(im, (64, 64)) for im in video])
        video = torch.tensor(video).permute((3, 0, 1, 2))
        video = torch.unsqueeze(video, 0)
        if self.is_cuda():
            video = video.cuda()

        return video

    def get_audio_tensor(self, audio: np.ndarray) -> torch.Tensor:
        audio = get_mfccs(audio, sample_rate=22050)
        audio = torch.tensor(audio)
        audio = torch.unsqueeze(audio, 0)
        if self.is_cuda():
            audio = audio.cuda()

        return audio

    def read(self):
        pass

    def stop(self):
        self.started = False
        self.thread.join()
