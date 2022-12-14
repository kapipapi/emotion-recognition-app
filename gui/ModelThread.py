import threading
import time

import numpy as np
import torch

from gui.CombinedSource import AVCapture
from gui.audio_preprocess import catch_audio_feature


class ModelThread:
    emotions = ["neutral/calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']

    last_faces = torch.zeros([15, 3, 224, 224])
    last_audio = torch.zeros([1, 181, 156])

    last_emotion = ""
    emotion_tensor = None

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
            start = time.time()
            (length_audio, audio_data), video = self.capture.read()

            if length_audio != self.capture.audio.n_samples or video.shape != (15, 3, 224, 224):
                print("No AV signals, waiting for buffor to fill-up.")
                time.sleep(1)
                continue

            audio = self.get_audio_tensor(audio_data)
            if audio.shape != torch.Size([1, 181, 156]):
                print("Wrong audio.shape: ", audio.shape, "want [1, 181, 156]")
                continue

            video = self.get_video_tensor(video)
            if video.shape != torch.Size([15, 3, 224, 224]):
                print("Wrong video.shape: ", video.shape, "want [15, 3, 224, 224]")
                continue

            self.model.eval()
            with torch.no_grad():
                if torch.equal(self.last_audio, audio) and torch.equal(self.last_faces, video):
                    print("WARNING: No face and audio detected, model does not predict.")
                else:
                    if torch.equal(self.last_audio, audio):
                        print("WARNING: No audio detected")
                        output = self.model(video=video)

                    elif torch.equal(self.last_faces, video):
                        print("WARNING: No face detected")
                        output = self.model(audio=audio)

                    else:
                        output = self.model(audio, video)

                    self.last_emotion = self.emotions[np.argmax(output.tolist())]
                    self.emotion_tensor = output.tolist()

            self.last_faces = video
            self.last_audio = audio

            print(time.time() - start, "seconds of model evaluation")

    def get_audio_tensor(self, audio: np.ndarray) -> torch.Tensor:
        audio = catch_audio_feature(audio, 22050)
        audio = torch.tensor(audio).float()
        audio = torch.unsqueeze(audio, 0)
        if self.device == torch.device('cuda'):
            audio = audio.cuda()
            self.last_faces = self.last_faces.cuda()

        return audio

    def get_video_tensor(self, video: [torch.Tensor]) -> [torch.Tensor]:
        if self.device == torch.device('cuda'):
            video = video.cuda()
            self.last_audio = self.last_audio.cuda()

        return video.float()

    def stop(self):
        self.started = False
        self.thread.join()

    def read(self):
        return self.last_emotion, self.emotion_tensor
