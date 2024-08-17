from flask import Flask, render_template, Response, request, session, redirect, url_for, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_socketio import SocketIO
import yt_dlp as youtube_dl
import time


model_object_detection = YOLO("yolov8x.pt")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')
stop_flag = False

# Global variables
camera = None
model = YOLO("yolov8x.pt")  # Load YOLOv8 model
total_detected_objects = 0
total_frames_processed = 0


class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self._preview = False
        self._flipH = False
        self._detect = False
        self._confidence = 75.0

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = int(value)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    def show(self, url):
        global total_detected_objects, total_frames_processed
        total_detected_objects = 0  # Reset total_detected_objects
        total_frames_processed = 0  # Reset total_frames_processed

        self._preview = False
        self._flipH = False
        self._detect = False
        self._confidence = 75.0

        if 'youtube.com' in url:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "best",
                "forceurl": True,
            }
            ydl = youtube_dl.YoutubeDL(ydl_opts)
            info = ydl.extract_info(url, download=False)
            url = info["url"]

        cap = cv2.VideoCapture(url)
        while True:
            if self._preview:
                if stop_flag:
                    print("Process Stopped")
                    return

                grabbed, frame = cap.read()
                if not grabbed:
                    break

                if self.flipH:
                    frame = cv2.flip(frame, 1)

                if self.detect:
                    total_frames_processed += 1
                    # Perform object detection
                    results = model_object_detection.predict(frame, conf=self._confidence/100)
                    frame, labels = results[0].plot()
                    list_labels = []
                    for label in labels:
                        confidence = label.split(" ")[-1]
                        label = " ".join(label.split(" ")[:-1])
                        list_labels.append(label)
                        list_labels.append(confidence)
                        socketio.emit('label', list_labels)
                    # Count total detected objects
                    total_detected_objects += len(labels)

                time.sleep(0.0)  # Adjust delay as needed

                frame = cv2.imencode(".jpg", frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            else:
                snap = np.zeros((1000, 1000), np.uint8)
                label = "Streaming Off"
                H, W = snap.shape
                font = cv2.FONT_HERSHEY_PLAIN
                color = (255, 255, 255)
                cv2.putText(snap, label, (W//2 - 100, H//2),
                            font, 2, color, 2)
                frame = cv2.imencode(".jpg", snap)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



VIDEO = VideoStreaming()

# Routes for the Flask application
@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('homepage.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    global stop_flag
    stop_flag = False
    if request.method == 'POST':
        url = request.form['url']
        session['url'] = url
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    url = session.get('url', None)
    if url is None:
        return redirect(url_for('homepage'))
    return Response(VIDEO.show(url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    global total_detected_objects, total_frames_processed
    accuracy = 0
    if total_frames_processed > 0:
        accuracy = (total_detected_objects / total_frames_processed) * 100
        # Cap the accuracy at 100%
        accuracy = min(accuracy, 100)
    return render_template('results.html', accuracy=accuracy, total_detected_objects=total_detected_objects)

@app.route('/get_accuracy', methods=['GET'])
def get_accuracy():
    global total_detected_objects, total_frames_processed
    accuracy = 0
    if total_frames_processed > 0:
        accuracy = (total_detected_objects / total_frames_processed) * 100
        # Cap the accuracy at 100%
        accuracy = min(accuracy, 100)
    return jsonify(accuracy=accuracy)

@app.route("/request_preview_switch")
def request_preview_switch():
    VIDEO.preview = not VIDEO.preview
    return "nothing"

@app.route("/request_flipH_switch")
def request_flipH_switch():
    VIDEO.flipH = not VIDEO.flipH
    return "nothing"

@app.route("/request_run_model_switch")
def request_run_model_switch():
    VIDEO.detect = not VIDEO.detect
    return "nothing"

@app.route('/update_slider_value', methods=['POST'])
def update_slider_value():
    slider_value = request.form['sliderValue']
    VIDEO.confidence = slider_value
    return 'OK'

@app.route('/stop_process')
def stop_process():
    global stop_flag, total_detected_objects, total_frames_processed
    stop_flag = True
    total_detected_objects = 0
    total_frames_processed = 0
    return 'Process Stop Request'



@socketio.on('connect')
def test_connect():
    print('Connected')

if __name__ == "__main__":
    socketio.run(app, debug=True)
