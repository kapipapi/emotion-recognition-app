import time
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.CombinedSource import AVCapture
from gui.ModelThread import ModelThread


class Booth:
    root: tk.Tk = None
    label: tk.Label = None

    seconds = 3.6
    capture: AVCapture = None

    model = None

    sample_freq_audio = 22050
    nb_samples_audio = int(seconds * sample_freq_audio)

    nb_samples_video = 50

    canvas = None
    toolbar = None
    figure = None
    line = None
    bar = None

    def __init__(self, plot_audio: bool, model: torch.nn.Module, device: torch.device):
        self.emotion_list_cache = None
        self.bar_canvas = None
        self.label_output = None
        self.plot_audio = plot_audio
        self.init_tk()
        self.init_capture()
        self.init_model(model, device)

        if self.plot_audio:
            self.init_audio_plot()
        else:
            self.init_bar_plot()

        self.print_info()
        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def init_tk(self):
        print("[!] Creating tkinter GUI")
        self.root = tk.Tk()

        self.label_output = tk.StringVar()

        self.root.geometry("640x660")

        self.root.title("AV Emotion Recognition")

        self.label = tk.Label(self.root, textvariable=self.label_output, compound="top",
                              font=("Calibri", 10))
        self.label.grid(row=0, column=0)

    def init_capture(self):
        print("[!] Async capture initialization")
        self.capture = AVCapture(
            sample_freq_audio=self.sample_freq_audio,
            n_samples_audio=self.nb_samples_audio,
            n_samples_video=self.nb_samples_video
        )
        self.capture.start()

    def init_model(self, model, device):
        print("[!] Model initialization")
        self.model = ModelThread(self.capture, model, device)
        self.model.start()

    def init_audio_plot(self):
        print("[!] Audio plot initialization")

        self.figure = plt.Figure(figsize=(8, 2), dpi=80)
        ax = self.figure.add_subplot()
        ax.set_ylim(-1, 1)

        time = np.linspace(-self.seconds, 0, num=self.nb_samples_audio)
        self.line, = ax.plot(time, [0] * self.nb_samples_audio)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0)

    def init_bar_plot(self):
        plt.style.use('ggplot')
        print("[!] Bar plot initialization")

        self.figure = plt.Figure(figsize=(8, 2), dpi=80)
        ax = self.figure.add_subplot()
        ax.set_ylim(0, 1)

        self.bar = ax.bar(["neutral/calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised'], [0] * 7,
                          color='#89CFF0', edgecolor='#818589', linewidth=1)

        self.bar_canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.bar_canvas.draw()
        self.bar_canvas.get_tk_widget().grid(row=1, column=0)

    def print_info(self):
        print("[i] Audio info:")
        print("[i] \t Sample frequency:", self.sample_freq_audio)
        print("[i] \t Sample duration:", self.seconds, "seconds")
        print("")
        print("[i] Video info:")
        print("[i] \t Sample count:", 15, "frames on net")
        print("[i] \t Sample buffer:", self.nb_samples_video, "frames on fifo")
        print("[i] \t Sample duration:", self.seconds, "seconds")
        print("")

    def draw_audio_plot(self, audio):
        self.line.set_ydata(audio)
        self.canvas.draw()
        self.canvas.flush_events()

    def draw_emotion_bar_plot(self):
        emotions = self.emotion_list_cache[0]
        emotions = np.exp(emotions) / sum(np.exp(emotions))

        for rect, h in zip(self.bar, emotions):
            rect.set_height(h)

        s = time.time()
        self.bar_canvas.draw()
        print(time.time() - s, "sekund suda")

    @staticmethod
    def cv2tk(image: np.ndarray) -> ImageTk:
        img = Image.fromarray(image, mode="RGB")
        return ImageTk.PhotoImage(image=img)

    def draw_video(self, cv_frame):
        video = self.cv2tk(cv_frame)
        self.label.imgtk = video
        self.label.configure(image=video)

    def audio_gui(self):
        if self.plot_audio:
            length, data_audio = self.capture.audio.read()
            if length == self.nb_samples_audio:
                self.draw_audio_plot(data_audio)

    def bar_plot(self):
        if not self.plot_audio:
            emotion, emotion_list = self.model.read()
            if emotion_list is not None and emotion_list is not self.emotion_list_cache:
                self.emotion_list_cache = emotion_list
                self.draw_emotion_bar_plot()

    def video_gui(self):
        if self.capture is not None:
            cv_frame = self.capture.video.read_live()
            if cv_frame is not None:
                self.draw_video(cv_frame)

    def update(self):
        self.audio_gui()
        self.video_gui()
        self.bar_plot()

        self.label_output.set(f"Detected emotion: {self.model.last_emotion}")

        self.root.after(5, self.update)

    def on_close(self):
        print("Ending processes")
        self.capture.stop()
        self.model.stop()
        del self.capture
        self.root.destroy()
        print("Processes closed")
