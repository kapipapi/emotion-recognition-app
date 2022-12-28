import threading
import time

import numpy as np
import torch

from gui.CombinedSource import AVCapture
from gui.audio_preprocess import catch_audio_feature


class ModelThread:
    emotions = ["neutral/calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']

    def __init__(self, capture: AVCapture, model: torch.nn.Module = None, device: torch.device = None):
        self.capture = capture

        self.started = False
        self.thread = None

        self.device = device
        self.model = model

        self.load_model()
        self.last_face_detection = torch.empty([15, 3, 224, 224])
        self.last_audio_detection = torch.empty([1, 181, 156])

    def load_model(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.model is not None

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

            if length_audio != self.capture.audio.n_samples or video.shape != (15, 3, 224, 224):
                time.sleep(1)
                continue

            audio = self.get_audio_tensor(audio_data)
            if audio.shape != torch.Size([1, 181, 156]):
                print("Wrong audio.shape: ", audio.shape, "want [1, 181, 156]")
                continue

            if video.shape != torch.Size([15, 3, 224, 224]):
                print("Wrong video.shape: ", video.shape, "want [15, 3, 224, 224]")
                continue

            self.model.eval()
            with torch.no_grad():
                if torch.equal(self.last_audio_detection, audio):
                    print("WARNING: No audio detected")
                    output = self.model(video=video.float())
                elif torch.equal(self.last_face_detection, video):
                    print("WARNING: No face detected")
                    output = self.model(audio=audio)
                else:
                    output = self.model(audio, video.float())

                emotion_index = np.argmax(output.tolist())
                print("Model output:", self.emotions[emotion_index])

            self.last_face_detection = video
            self.last_audio_detection = audio

    def get_audio_tensor(self, audio: np.ndarray) -> torch.Tensor:
        audio = catch_audio_feature(audio, 22050)
        audio = torch.tensor(audio).float()
        audio = torch.unsqueeze(audio, 0)
        if self.device == torch.device('cuda'):
            audio = audio.cuda()

        return audio

    def stop(self):
        self.started = False
        self.thread.join()
