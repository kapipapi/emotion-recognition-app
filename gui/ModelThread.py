import threading
import time

import cv2
import numpy as np
import torch

from gui.CombinedSource import AVCapture
from gui.audio_preprocess import catch_audio_feature


class ModelThread:
    emotions = ["neutral", "calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']

    def __init__(self, capture: AVCapture, model: torch.nn.Module = None, device: torch.device = None):
        self.capture = capture

        self.started = False
        self.thread = None

        self.device = device
        self.model = model

        self.load_model()
        # self.warm_model()

    def load_model(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.model is not None

        # self.model.to(self.device)
        # if self.is_cuda():
        #     self.model.cuda(self.device)
        #
        # self.model.eval()

    def is_cuda(self):
        return self.device == torch.device('cuda')

    def warm_model(self):
        print("[!] Model warmup")
        audio_rand = np.random.random([int(22050 * 3.6)]).astype('f')
        audio_rand = self.get_audio_tensor(audio_rand)

        video_rand = np.random.random([15, 224, 224, 3]).astype('f')
        video_rand = self.get_video_tensor(video_rand)

        self.model.eval()
        with torch.no_grad():
            video_rand = video_rand.permute(0, 2, 1, 3, 4)
            video_rand = video_rand.reshape(video_rand.shape[0] * video_rand.shape[1], video_rand.shape[2],
                                            video_rand.shape[3], video_rand.shape[4])
            video_rand = video_rand.float()

            self.model(audio_rand, video_rand)

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

            # test = []
            # for f in video:
                # test.append(np.array(f.permute(1, 2, 0).int().tolist()))

            # buf = cv2.hconcat(test)

            audio = self.get_audio_tensor(audio_data)
            if audio.shape != torch.Size([1, 181, 156]):
                print("wrongaudio.shape:", audio.shape, "want [1, 181, 156]")
                continue

            if video.shape != torch.Size([15, 3, 224, 224]):
                print("wrong video.shape:", video.shape, "want [15, 3, 224, 224]")
                continue

            with torch.no_grad():
                output = self.model(audio, video.cuda().float())
                print(output)
                output = np.argmax(output.tolist())

            print("model output:", self.emotions[output])

    def get_video_tensor(self, video) -> torch.Tensor:
        video = np.array([cv2.resize(im, (224, 224)) for im in video])
        video = torch.tensor(video).permute((3, 0, 1, 2))
        video = torch.unsqueeze(video, 0)
        if self.is_cuda():
            video = video.cuda()

        return video

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
        self.started = False
        self.thread.join()
