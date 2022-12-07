import cv2
import time

import numpy as np
import torch
import threading
from collections import deque
from facenet_pytorch import MTCNN

from gui.SensorSource import SensorSource


class VideoSource(SensorSource):
    """Object for video using OpenCV."""
    last_timestamp = 0

    def __init__(self, src=0, nb_samples=7, sample_freq=2):
        """Initialise video capture."""
        # width=640, height=480
        super().__init__()
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        _, self.frame = self.cap.read()

        self.nb_samples = nb_samples
        self.sample_freq = sample_freq
        self.image_live = self.frame
        self.fifo_image = deque(maxlen=self.nb_samples)
        self.fifo_timestamp_ns = deque(maxlen=self.nb_samples)

        self.started = False
        self.read_lock = threading.Lock()

        self.buffer_calculated_fps = 0

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(image_size=(480, 640), device=self.device)
        self.mtcnn.to(self.device)

    def calculate_fps(self, timestamp):
        last_timestamp = 0
        if len(self.fifo_timestamp_ns) > 0:
            last_timestamp = self.fifo_timestamp_ns[-1]

        diff = timestamp - last_timestamp
        if diff == 0:
            return 0

        return 10 ** 9 / diff

    def update(self):
        """Update based on new video data."""
        while self.started:
            grabbed, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self.read_lock:
                self.image_live = frame

            timestamp = time.time_ns()
            fps = self.calculate_fps(timestamp)
            if fps >= self.sample_freq * 1.1:
                continue

            img_tensor = torch.tensor(frame)
            img_tensor = img_tensor.to(self.device)
            bbox = self.mtcnn.detect(img_tensor)
            if bbox[0] is not None:
                bbox = bbox[0][0]
                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox
                face_cropped = frame[y1:y2, x1:x2, :]
                face_cropped = cv2.resize(face_cropped, (224, 224))

                with self.read_lock:
                    if grabbed:
                        self.buffer_calculated_fps = fps
                        self.fifo_image.append(face_cropped)
                        self.fifo_timestamp_ns.append(timestamp)

    def read(self) -> np.ndarray:
        """Read video."""
        with self.read_lock:
            return np.asarray(self.fifo_image)

    def read_live(self):
        """Read live video feed."""
        with self.read_lock:
            return self.image_live

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


if __name__ == "__main__":
    vs = VideoSource()
    vs.start()
    while True:
        x = vs.read_live()
        cv2.imshow("test", x)
        if cv2.waitKey(10) == ord('q'):
            break
    vs.stop()
