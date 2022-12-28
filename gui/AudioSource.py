import audioop
import math
import threading
from collections import deque

import numpy as np
import pyaudio

from gui import audio_format
from gui.SensorSource import SensorSource


class AudioSource(SensorSource):
    """Object for audio using alsaaudio."""

    def __init__(self, sample_freq=44100, nb_samples=65536):
        """Initialise audio capture."""
        super().__init__()

        self.n_samples = nb_samples
        self.sample_freq = sample_freq
        self.CHUNK = 1024
        self.p = pyaudio.PyAudio()
        self.inp = self.p.open(format=pyaudio.paInt16, channels=1, rate=sample_freq, input=True,
                               frames_per_buffer=self.CHUNK)
        self.read_lock = threading.Lock()

        # Create a FIFO structure for the data
        self._s_fifo = deque(maxlen=self.n_samples)
        self.started = False
        self.read_lock = threading.Lock()

        rel = self.sample_freq / self.CHUNK
        self.slid_window = deque(maxlen=int(rel))
        self.audio_threshold = 2000

    def update(self):
        """Update based on new audio data."""
        while self.started:
            data = self.inp.read(self.CHUNK, exception_on_overflow=False)
            self.slid_window.append(math.sqrt(abs(audioop.avg(data, 4))))

            if len(data) > 0 and sum([x > self.audio_threshold for x in self.slid_window]) > 0:
                with self.read_lock:
                    data_float = audio_format.byte_to_float(data)
                    self._s_fifo.extend(data_float)
            else:
                continue

    def read(self):
        """Read audio."""
        with self.read_lock:
            return len(self._s_fifo), np.asarray(self._s_fifo)
