# Audio-video asynchronous GUI

1. Clone repo
2. make env `python3 -m venv venv`
3. activate `source venv/bin/activate`
4. Install `pyaudio` requirements https://pypi.org/project/PyAudio/
5. `pip install -r requirements.txt`

# integrate model
1. Go to root of project "emotion-recognition-app"
2. Clone model for example [this](https://github.com/TheMatiwoz/av-emotion-recognition) (change name bc of minus sigh)
3. `git clone git@github.com:TheMatiwoz/av-emotion-recognition.git model`
3. In `main.py` import model and pass it to Booth class
4. Try not to push it to this repo :)