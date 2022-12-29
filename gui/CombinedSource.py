import numpy as np

from gui.AudioSource import AudioSource
from gui.VideoSource import VideoSource


class CombinedSource:
    def __init__(self):
        self.audio = None
        self.video = None

    def start(self):
        if self.audio is not None:
            self.audio.start()

        if self.video is not None:
            self.video.start()

    def read(self) -> ((int, np.array), np.array):
        data_audio, data_video = None, None

        if self.audio is not None:
            data_audio = self.audio.read()

        if self.video is not None:
            data_video = self.video.read()

        return data_audio, data_video

    def stop(self):
        if self.audio is not None:
            self.audio.stop()

        if self.video is not None:
            self.video.stop()

    def __del__(self):
        if self.video is not None:
            self.video.cap.release()

    def __exit__(self, exec_type, exc_value, traceback):
        if self.video is not None:
            self.video.cap.release()


class AVCapture(CombinedSource):
    def __init__(self, n_samples_video, sample_freq_audio, n_samples_audio):
        super().__init__()
        self.audio = AudioSource(sample_freq_audio, n_samples_audio)
        self.video = VideoSource(0, n_samples_video)
