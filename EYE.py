from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


# ---------------- Eye Aspect Ratio Function ----------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# ---------------- Argument Parser ----------------
ap = argparse.ArgumentParser()
ap.add_argument(
    "-p",
    "--shape-predictor",
    required=True,
    help='E:\project\DRY EYES\shape_predictor_68_face_landmarks.dat'
)
args = vars(ap.parse_args())


# ---------------- Constants ----------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0


# ---------------- Load Dlib Models ----------------
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# ---------------- Start Video Stream ----------------
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)


# ---------------- Main Loop ----------------
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Blink detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        # Display info
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------- Cleanup ----------------
cv2.destroyAllWindows()
vs.stop()

#TO run Use this comment =  python EYE.py -p "D:\Repo\2025_2026\Main\LBS\Dry Eyes Detection\EYE_Detection\shape_predictor_68_face_landmarks.dat"


