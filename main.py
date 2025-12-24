import cv2
import face_recognition
import os
import csv
from datetime import datetime

# ---------------- CONFIG ---------------- #

KNOWN_FACES_DIR = "image"
ATTENDANCE_FILE = "attendance.csv"

# Late threshold (9:00 AM)
LATE_TIME = datetime.strptime("09:00:00", "%H:%M:%S").time()

# ---------------------------------------- #

known_encodings = []
known_names = []

# Load known faces
for file in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{file}")
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(file.split(".")[0])

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Status"])

# Start webcam
video = cv2.VideoCapture(0)

# Prevent duplicate attendance in same run
marked = set()

while True:
    success, frame = video.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

            if name not in marked:
                now = datetime.now()

                status = "Late" if now.time() > LATE_TIME else "Present"

                with open(ATTENDANCE_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        status
                    ])

                marked.add(name)

        # Draw face box and name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
