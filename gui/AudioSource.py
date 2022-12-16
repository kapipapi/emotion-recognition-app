import logging
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
        # Initialise audio
        super().__init__()

        self.n_samples = nb_samples

        self.CHUNK = 1024
        self.p = pyaudio.PyAudio()
        self.inp = self.p.open(format=pyaudio.paInt16,
                               channels=1,
                               rate=sample_freq,
                               input=True,
                               frames_per_buffer=self.CHUNK)
        self.read_lock = threading.Lock()

        # Create a FIFO structure for the data
        self._s_fifo = deque(maxlen=self.n_samples)
        self.started = False
        self.read_lock = threading.Lock()

    def update(self):
        """Update based on new audio data."""
        while self.started:
            data = self.inp.read(self.CHUNK, exception_on_overflow=False)

            if len(data) > 0:
                with self.read_lock:
                    data_float = audio_format.byte_to_float(data)
                    self._s_fifo.extend(data_float)
            else:
                logging.error(f'Sampler error occur')

    def read(self):
        """Read audio."""
        with self.read_lock:
            return len(self._s_fifo), np.asarray(self._s_fifo)
