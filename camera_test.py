from collections import deque

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from model.models.model import MultimodalModelTFusion

cap = cv2.VideoCapture("../Actor_01/01-01-03-01-02-01-01.mp4")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultimodalModelTFusion()
model.load_state_dict(torch.load("../weights/model_76_final.pth", map_location=device))
model.to(device)

mtcnn = MTCNN(device=device)
mtcnn.to(device)

faces = deque(maxlen=15)

emotions = ["neutral", "calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']


def run_model(video_data):
    clip = []
    for i in range(np.shape(video_data)[0]):
        clip.append(torch.tensor(video_data[i]))

    clip = torch.stack(clip, 0).permute(0, 3, 1, 2)

    print(clip.shape)

    clip = clip.cuda().float()

    output = model([], clip)

    print(emotions[np.argmax(output.tolist())])


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    im_rgb = torch.tensor(frame)
    im_rgb = im_rgb.to(device)

    bbox = mtcnn.detect(im_rgb)
    try:
        if bbox[0] is not None:
            bbox = bbox[0][0]
            bbox = [round(x) for x in bbox]
            x1, y1, x2, y2 = bbox

            im = frame[y1:y2, x1:x2, :]
            im = cv2.resize(im, (224, 224))
            faces.append(im)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    except:
        continue

    if len(faces) == 15:
        run_model(faces)

    cv2.imshow("test", frame)
    if cv2.waitKey(1) == ord('q'):
        exit(0)
