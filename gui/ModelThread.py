import threading
import time

import cv2
import numpy as np
import torch

from gui.CombinedSource import AVCapture
from gui.audio_preprocess import catch_audio_feature


class ModelThread:
    emotions = ["neutral", "calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']
    output_csv = []

    def __init__(self, capture: AVCapture, model: torch.nn.Module = None, device: torch.device = None):
        self.capture = capture

        self.started = False
        self.thread = None

        self.device = device
        self.model = model

        self.load_model()

    def load_model(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.model is not None

    def is_cuda(self):
        return self.device == torch.device('cuda')

    def start(self):
        if self.started:
            return None
        self.started = True

        print("Model thread starting")

        self.thread = threading.Thread(
            target=self.update,
            args=()
        )
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (length_audio, audio_data), video = self.capture.read()

            if video is None:
                continue

            if length_audio != self.capture.audio.n_samples or video.shape != (15, 3, 224, 224):
                time.sleep(1)
                continue

            audio = self.get_audio_tensor(audio_data)
            if audio.shape != torch.Size([1, 181, 156]):
                print("wrongaudio.shape:", audio.shape, "want [1, 181, 156]")
                continue

            if video.shape != torch.Size([15, 3, 224, 224]):
                print("wrong video.shape:", video.shape, "want [15, 3, 224, 224]")
                continue

            # self.model.eval()
            with torch.no_grad():
                output = self.model(audio, video.cuda().float())
                output = output.tolist()

                output_norm = np.exp(output) / np.sum(np.exp(output), axis=1)
                self.output_csv.append(output_norm[0])

                emotion_index = np.argmax(output)

            print("model output:", self.emotions[emotion_index])

    def get_audio_tensor(self, audio: np.ndarray) -> torch.Tensor:
        audio = catch_audio_feature(audio, 22050)
        audio = torch.tensor(audio).float()
        audio = torch.unsqueeze(audio, 0)
        if self.is_cuda():
            audio = audio.cuda()

        return audio

    def read(self):
        pass

    def stop(self):
        try:
            np.savetxt("text.csv", self.output_csv, delimiter=",")
        except:
            print("error with saving output to csv")
        self.started = False
        self.thread.join()
