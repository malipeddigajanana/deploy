import face_recognition
import cv2
import numpy as np
import threading
import pickle

pickle_name = "face_encodings_custom.pickle"  # the name of the pickle file where the encodings are stored

# Load encodings from pickle file
data_encoding = pickle.load(open(pickle_name, "rb"))
known_face_encodings = data_encoding["encodings"]
known_face_names = data_encoding["names"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def recognize_faces(frame):
    global face_locations, face_encodings, face_names, process_this_frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize the frame to reduce size
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
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

    process_this_frame = not process_this_frame

def video_stream():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        recognize_faces(frame)
        draw_faces(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def draw_faces(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2  # Adjust to new size
        right *= 2  # Adjust to new size
        bottom *= 2  # Adjust to new size
        left *= 2  # Adjust to new size
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

video_thread = threading.Thread(target=video_stream)
video_thread.start()
