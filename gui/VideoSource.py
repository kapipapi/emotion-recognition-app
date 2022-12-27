import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from gui.SensorSource import SensorSource


class VideoSource(SensorSource):
    """Object for video using OpenCV."""

    def __init__(self, src, n_samples):
        """Initialise video capture."""
        # width=640, height=480
        super().__init__()
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        _, self.frame = self.cap.read()

        self.n_samples = n_samples
        self.image_live = self.frame
        self.fifo_image = deque(maxlen=self.n_samples)
        self.timestamps = deque(maxlen=self.n_samples)

        self.started = False
        self.read_lock = threading.Lock()

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(device=self.device)
        self.mtcnn.to(self.device)

    def update(self):
        """Update based on new video data."""
        while self.started:
            grabbed, frame = self.cap.read()
            self.timestamps.append(time.time())

            if not grabbed:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            temp = frame[:, :, -1]
            im_rgb = frame.copy()
            im_rgb[:, :, -1] = im_rgb[:, :, 0]
            im_rgb[:, :, 0] = temp
            im_rgb = torch.tensor(im_rgb)
            im_rgb = im_rgb.to(self.device)

            bbox = self.mtcnn.detect(im_rgb)
            if bbox[0] is not None:
                bbox = bbox[0][0]
                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox

                face_cropped = frame[y1:y2, x1:x2, :]

                if 0 in face_cropped.shape:
                    continue

                face_cropped = cv2.resize(face_cropped, (224, 224))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 150), 3)

                with self.read_lock:
                    self.fifo_image.append(face_cropped)
                    self.image_live = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                with self.read_lock:
                    self.image_live = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read(self, n: int = 15) -> np.ndarray:
        """Read video."""
        with self.read_lock:
            fifo_recent = self.fifo_image

            if len(fifo_recent) != self.n_samples:
                return np.array([])

            print("differance", self.timestamps[-1] - self.timestamps[0])

            clip = []
            idx = np.linspace(0, len(fifo_recent) - 1, n, dtype='int')
            for i in idx:
                frame = fifo_recent[i]
                clip.append(torch.tensor(frame))

            # shape 15, 224, 224, 3

            clip = torch.stack(clip, 0).permute(0, 3, 1, 2)

            # shape 15, 3, 224, 224

            return clip

    def read_live(self):
        """Read live video feed."""
        with self.read_lock:
            return self.image_live

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


if __name__ == "__main__":
    vs = VideoSource(src=0, n_samples=108)
    vs.start()
    while True:
        x = vs.read_live()
        cv2.imshow("test", x)
        if cv2.waitKey(10) == ord('q'):
            break
    vs.stop()
