import face_recognition
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import dlib
import cv2
import csv
import os
from datetime import datetime
import time

# --- 1. DIRECTORY SETUP ---
DATA_DIR = "user_sessions"
PROFILES_DIR = "face_profiles"
GRAPHS_DIR = "user_graphs"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

for d in [DATA_DIR, PROFILES_DIR, GRAPHS_DIR]:
    if not os.path.exists(d): os.makedirs(d)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def load_known_faces():
    encs, names = [], []
    for f in os.listdir(PROFILES_DIR):
        if f.endswith(".npy"):
            names.append(f.replace(".npy", ""))
            encs.append(np.load(os.path.join(PROFILES_DIR, f)))
    return encs, names

# --- 2. INITIALIZATION ---
print("[INFO] Initializing System...")
print("[WARNING] Running in software-only mode.") 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
known_encs, known_names = load_known_faces()

# --- 3. REGISTRATION GATE (For the very first time only) ---
vs = VideoStream(src=0).start()
time.sleep(2.0)

if not known_names:
    print("\n" + "="*40)
    print("[USER REGISTRATION] No Profiles Found.")
    print("="*40)
    
    reg_name = input("Enter Name for Registration: ").strip()
    
    print(f"\nHi {reg_name}, looking for your face now...")
    while True:
        frame = vs.read()
        if frame is None: continue
        
        frame = cv2.resize(frame, (640, 480))
        
        cv2.putText(frame, "Registration: Please face the camera", (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.55, (0, 0, 0), 2)
        cv2.imshow("Dry Eye IoT System", frame)
        cv2.waitKey(1) 
        
        temp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        temp_gray = cv2.equalizeHist(temp_gray)
        rects = detector(temp_gray, 1) 
        
        if len(rects) > 0:
            rgb_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_temp, model="hog")
            if face_locs:
                enc = face_recognition.face_encodings(rgb_temp, face_locs)[0]
                np.save(os.path.join(PROFILES_DIR, f"{reg_name}.npy"), enc)
                print(f"[SUCCESS] Registered {reg_name}. Opening System...")
                known_encs, known_names = load_known_faces()
                
                cv2.destroyWindow("Dry Eye IoT System")
                break
        print("Searching for face... please face the camera", end="\r")
        time.sleep(0.1)

# Variables
current_user = "Scanning..."
total_blinks = 0
COUNTER = 0
last_id_check = 0
session_start = None
csv_path = None
session_id = 0
humidifier_intensity = 50 
auto_boost_triggered = False 
bpm = 0.0
status_text = "Scanning"
bpm_history = [] 

# --- 4. MONITORING LOOP ---
while True:
    frame = vs.read()
    if frame is None: continue
    frame = cv2.resize(frame, (640, 480))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    rects = detector(gray, 0) 
    
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if time.time() - last_id_check > 2.0:
        if rects:
            face_locs = face_recognition.face_locations(rgb)
            face_encs = face_recognition.face_encodings(rgb, face_locs)
            
            if face_encs:
                matches = face_recognition.compare_faces(known_encs, face_encs[0], 0.6)
                if True in matches:
                    match_name = known_names[matches.index(True)]
                    if match_name != current_user:
                        current_user = match_name
                        total_blinks = 0
                        session_start = time.time()
                        auto_boost_triggered = False
                        bpm_history = []
                        csv_path = os.path.join(DATA_DIR, f"{current_user}.csv")
                        
                        intensity_path = os.path.join(DATA_DIR, f"{current_user}_intensity.txt")
                        if os.path.exists(intensity_path):
                            with open(intensity_path, 'r') as f:
                                try:
                                    humidifier_intensity = int(f.read().strip())
                                except ValueError:
                                    humidifier_intensity = 50
                        else:
                            humidifier_intensity = 50
                        
                        file_exists = os.path.exists(csv_path)
                        if not file_exists:
                            session_id = 1
                            with open(csv_path, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(["Session_ID", "Timestamp", "Total_Blinks", "BPM", "Status"])
                        else:
                            with open(csv_path, 'r') as f:
                                reader = list(csv.reader(f))
                                if len(reader) > 1:
                                    try:
                                        session_id = int(reader[-1][0]) + 1
                                    except: session_id = 1
                                else: session_id = 1
        else:
            if current_user == "Scanning...":
                print("[WARN] No face detected. Searching...", end="\r")
        last_id_check = time.time()

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        ear = (eye_aspect_ratio(shape[36:42]) + eye_aspect_ratio(shape[42:48])) / 2.0

        if ear < 0.23: 
            COUNTER += 1
        else:
            if COUNTER >= 2:
                total_blinks += 1
                if csv_path and session_start:
                    bpm_history.append(bpm)
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([session_id, datetime.now().strftime("%H:%M:%S"), total_blinks, round(bpm, 1), status_text])
            COUNTER = 0

    status_color = (0, 0, 0)
    if session_start and current_user != "Scanning...":
        elapsed = time.time() - session_start
        dur_min = elapsed / 60
        
        if dur_min > 0.02: 
            bpm = total_blinks / dur_min
        else:
            bpm = 20.0 

        if bpm >= 20: 
            status_text, status_color = "Healthy", (0, 255, 0)
            auto_boost_triggered = False
        elif 10 <= bpm < 20: 
            status_text, status_color = "Moderate", (0, 255, 255)
            auto_boost_triggered = False
        else: 
            status_text, status_color = "Severe", (0, 0, 255)
            if bpm < 5 and elapsed > 20 and not auto_boost_triggered:
                humidifier_intensity = min(100, humidifier_intensity + 10)
                
                with open(os.path.join(DATA_DIR, f"{current_user}_intensity.txt"), 'w') as f:
                    f.write(str(humidifier_intensity))
                    
                print(f"\n[AUTO] Extreme dryness detected. Level: {humidifier_intensity}%")
                auto_boost_triggered = True

    FONT = cv2.FONT_HERSHEY_COMPLEX
    H_SIZE, V_SIZE, STEP = 0.55, 0.50, 25
    BLACK = (0, 0, 0)
    
    key = cv2.waitKey(1) & 0xFF
    
    # --- NEW: PRESS 'r' TO REGISTER A NEW USER MID-SESSION ---
    if key == ord('r'):
        cv2.destroyAllWindows() # Safely hide window so it doesn't crash
        print("\n" + "="*40)
        new_name = input("[NEW REGISTRATION] Enter Name: ").strip()
        if new_name:
            print(f"Capturing face for {new_name}... Please look at the camera.")
            # Grab a quick frame to save the new face
            fresh_frame = vs.read()
            if fresh_frame is not None:
                rgb_fresh = cv2.cvtColor(fresh_frame, cv2.COLOR_BGR2RGB)
                face_locs_fresh = face_recognition.face_locations(rgb_fresh, model="hog")
                if face_locs_fresh:
                    new_enc = face_recognition.face_encodings(rgb_fresh, face_locs_fresh)[0]
                    np.save(os.path.join(PROFILES_DIR, f"{new_name}.npy"), new_enc)
                    print(f"[SUCCESS] {new_name} registered! Resuming system...")
                    known_encs, known_names = load_known_faces() # Reload database
                else:
                    print("[ERROR] Could not see a face. Registration failed. Resuming...")
        print("="*40 + "\n")

    elif key == ord('h') or key == ord('l'):
        if key == ord('h'): 
            humidifier_intensity = min(100, humidifier_intensity + 5)
        elif key == ord('l'): 
            humidifier_intensity = max(0, humidifier_intensity - 5)
            
        if current_user != "Scanning...":
            with open(os.path.join(DATA_DIR, f"{current_user}_intensity.txt"), 'w') as f:
                f.write(str(humidifier_intensity))

    cv2.putText(frame, "USER:", (15, 30), FONT, H_SIZE, BLACK, 2)
    cv2.putText(frame, str(current_user), (85, 30), FONT, V_SIZE, BLACK, 1)
    cv2.putText(frame, "BPM:", (15, 30 + STEP), FONT, H_SIZE, BLACK, 2)
    cv2.putText(frame, f"{bpm:.1f}", (70, 30 + STEP), FONT, V_SIZE, BLACK, 1)
    cv2.putText(frame, "LEVEL:", (15, 30 + (STEP * 2)), FONT, H_SIZE, BLACK, 2)
    cv2.putText(frame, f"{humidifier_intensity}%", (95, 30 + (STEP * 2)), FONT, V_SIZE, BLACK, 1)
    cv2.putText(frame, "STATUS:", (15, 30 + (STEP * 3)), FONT, H_SIZE, BLACK, 2)
    cv2.circle(frame, (115, 30 + (STEP * 3) - 5), 6, status_color, -1)
    cv2.putText(frame, status_text, (135, 30 + (STEP * 3)), FONT, V_SIZE, status_color, 2 if status_text != "Scanning" else 1)

    cv2.imshow("Dry Eye IoT System", frame)
    if key == ord('q'): break

cv2.destroyAllWindows(); vs.stop()

# --- 5. GRAPH DISPLAY AFTER QUIT ---
if bpm_history and current_user != "Scanning...":
    print("\n[INFO] Generating session graph...")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(bpm_history, color='blue', marker='o', linestyle='-')
    plt.title(f"Blink Rate Analysis: {current_user} (Session {session_id})")
    plt.xlabel("Blink Events")
    plt.ylabel("BPM")
    plt.grid(True)
    
    plt.axhline(y=20, color='green', linestyle='--', label="Healthy Threshold (20 BPM)")
    plt.axhline(y=10, color='red', linestyle='--', label="Severe Threshold (10 BPM)")
    plt.legend()
    
    graph_path = os.path.join(GRAPHS_DIR, f"{current_user}_{session_id}.png")
    plt.savefig(graph_path)
    print(f"[SUCCESS] Graph saved to {graph_path}. Displaying window now...")
    plt.show()