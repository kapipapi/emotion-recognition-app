import threading
from collections import deque

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from gui.SensorSource import SensorSource


class VideoSource(SensorSource):
    """Object for video using OpenCV."""
    last_timestamp = 0

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

        self.started = False
        self.read_lock = threading.Lock()

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(image_size=(480, 640), device=self.device)
        self.mtcnn.to(self.device)

    def update(self):
        """Update based on new video data."""
        while self.started:
            grabbed, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self.read_lock:
                self.image_live = frame

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
                        self.fifo_image.append(face_cropped)

    def read(self, n: int = 15) -> np.ndarray:
        """Read video."""
        with self.read_lock:
            fifo_recent = np.asarray(self.fifo_image)

        if len(fifo_recent) != self.n_samples:
            return np.array([])

        faces = []
        for frame in fifo_recent[::self.n_samples // n + 1]:
            faces.append(frame)

        faces.append(fifo_recent[-1])
        return np.asarray(faces)

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
