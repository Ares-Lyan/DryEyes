import face_recognition
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import csv
import os
import serial  # Added for ESP32
from datetime import datetime

# ----------------- Functions -----------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def generate_visual_report(log_filename, user_name):
    timestamps, blink_counts = [], []
    try:
        if not os.path.exists(log_filename): return
        with open(log_filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                time_obj = datetime.strptime(row['Timestamp'], "%H:%M:%S.%f")
                if not timestamps: start_time = time_obj
                timestamps.append((time_obj - start_time).total_seconds())
                blink_counts.append(int(row['Total_Blinks']))
        
        if len(timestamps) < 2: return
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, blink_counts, marker='o', color='b')
        plt.title(f"Blink Pattern Analysis: {user_name}")
        plt.xlabel("Seconds into Session")
        plt.ylabel("Cumulative Blink Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(log_filename.replace(".csv", ".png"))
        plt.show()
    except Exception as e: print(f"[ERROR] Graphing failed: {e}")

def register_new_user(vs):
    print("\n" + "="*40)
    user_name = input("Enter User Name for this session: ").strip() or "User"
    print(f"[INFO] Capturing Face ID for {user_name}...")
    time.sleep(2.0)
    frame = vs.read()
    rgb_reg = cv2.cvtColor(imutils.resize(frame, width=450), cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_reg)
    if len(boxes) > 0:
        encoding = face_recognition.face_encodings(rgb_reg, boxes)[0]
        log_name = f"{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(log_name, mode='w', newline='') as file:
            csv.writer(file).writerow(["Timestamp", "Total_Blinks", "EAR_Value"])
        print(f"[SUCCESS] {user_name} registered.")
        return encoding, user_name, log_name
    return None, None, None

# ----------------- Setup -----------------
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True)
args = vars(ap.parse_args())

# ESP32 Connection
try:
    ser = serial.Serial('COM3', 115200, timeout=1) # Update port if needed
    print("[INFO] ESP32 Connected.")
except:
    ser = None
    print("[WARNING] Running in software-only mode.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
time.sleep(2.0)

master_encoding, current_user, log_filename = register_new_user(vs)
while master_encoding is None:
    master_encoding, current_user, log_filename = register_new_user(vs)

EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES = 0.25, 3
COUNTER, TOTAL = 0, 0
session_start_time = datetime.now()

# ----------------- Monitoring Loop -----------------
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    rects = detector(gray, 0)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(rects) > 1:
        cv2.putText(frame, "WARNING: MULTIPLE PEOPLE DETECTED", (10, 450), 1, 0.7, (0,0,255), 2)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([master_encoding], encoding)
        
        if matches[0]:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            shape = predictor(gray, dlib.rectangle(left, top, right, bottom))
            shape = face_utils.shape_to_np(shape)
            ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    curr_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    with open(log_filename, mode='a', newline='') as f:
                        csv.writer(f).writerow([curr_time, TOTAL, f"{ear:.2f}"])
                COUNTER = 0

            # NEW: Real-time Analysis & IoT Trigger
            duration_min = (datetime.now() - session_start_time).total_seconds() / 60
            bpm = TOTAL / duration_min if duration_min > 0.1 else 15
            
            if bpm < 10: status, code = "SEVERE", "2"
            elif bpm < 15: status, code = "MODERATE", "1"
            else: status, code = "HEALTHY", "0"

            if ser: ser.write(code.encode())

            cv2.putText(frame, f"User: {current_user} | BPM: {bpm:.1f}", (10, 30), 1, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Status: {status}", (10, 60), 1, 0.7, (0, 255, 255), 2)

    cv2.imshow("IoT Dry Eye Care System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        generate_visual_report(log_filename, current_user)
        new_enc, new_name, new_log = register_new_user(vs)
        if new_enc:
            master_encoding, current_user, log_filename = new_enc, new_name, new_log
            TOTAL = 0
            session_start_time = datetime.now()
    elif key == ord("q"): break

cv2.destroyAllWindows(); vs.stop()
if ser: ser.close()
generate_visual_report(log_filename, current_user)