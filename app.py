# app.py
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import threading
import pickle

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load sample pictures and learn how to recognize them
pickle_name = "face_encodings_custom.pickle"  # the name of the pickle file where are stored the encodings
max_width = 400  # reduce the max_width for smaller output video size

# Load encodings from pickle file
data_encoding = pickle.loads(open(pickle_name, "rb").read())
known_face_encodings = data_encoding["encodings"]
known_face_names = data_encoding["names"]

def recognize_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    return face_locations, face_names

def video_stream():
    while True:
        success, frame = camera.read()
        if not success:
            break

        face_locations, face_names = recognize_faces(frame)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Resize frame before sending
        if max_width is not None:
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * (max_width / frame.shape[1]))))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
