import face_recognition
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import dlib
import cv2
import csv
import os
import serial 
from datetime import datetime
import time

# --- NEW: FIREBASE IMPORTS ---
import firebase_admin
from firebase_admin import credentials, db

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

# --- FIREBASE SETUP ---
try:
    cred = credentials.Certificate(r"C:\DryEyes-main\dry-eyes-detection-firebase-adminsdk-fbsvc-29cfd48159.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://dry-eyes-detection-default-rtdb.asia-southeast1.firebasedatabase.app"
    })
    firebase_enabled = True
    print("[SUCCESS] Connected to Firebase Dashboard!")
except Exception as e:
    firebase_enabled = False
    print(f"[WARNING] Firebase setup failed. Check your JSON file. Running locally. Error: {e}")

# --- ARDUINO SETUP ---
try:
    arduino = serial.Serial('COM4', 115200, timeout=1) 
    time.sleep(2.0)
    print("[SUCCESS] Connected to Arduino")
except Exception as e:
    arduino = None
    print("[WARNING] Arduino not found on Running in software-only mode.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
known_encs, known_names = load_known_faces()

# --- 3. REGISTRATION GATE ---
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
                largest_loc = max(face_locs, key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))
                enc = face_recognition.face_encodings(rgb_temp, [largest_loc])[0]
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
dry_eye_events = 0 
COUNTER = 0
last_id_check = 0
session_start = None
csv_path = None
session_id = 0
humidifier_intensity = 50 
bpm = 0.0
status_text = "Scanning"
bpm_history = [] 

# NEW: Sensor Variables
current_temp = 0.0
current_humidity = 0.0

last_boost_time = 0 
last_firebase_update = 0 

FATIGUE_FRAMES = 15  
fatigue_alert = False

# --- 4. MONITORING LOOP ---
while True:
    frame = vs.read()
    if frame is None: continue
    frame = cv2.resize(frame, (640, 480))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    rects = detector(gray, 0) 
    main_rect = None
    
    if len(rects) > 0:
        main_rect = max(rects, key=lambda rect: rect.width() * rect.height())
        (x, y, w, h) = face_utils.rect_to_bb(main_rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # --- NEW: READ SENSOR DATA FROM ARDUINO ---
    if arduino is not None and arduino.in_waiting > 0:
        try:
            # Read the line, decode it, and remove extra spaces
            serial_data = arduino.readline().decode('utf-8').strip()
            
            # Look for our exact format: "T:33.5,H:64.2"
            if "T:" in serial_data and "H:" in serial_data:
                parts = serial_data.split(',')
                current_temp = float(parts[0].split(':')[1])
                current_humidity = float(parts[1].split(':')[1])
        except Exception as e:
            pass # Ignore any scrambled data during startup

    # 1. SMART IDENTITY LOCK
    if time.time() - last_id_check > 2.0:
        if main_rect is not None:
            face_locs = face_recognition.face_locations(rgb)
            if face_locs:
                largest_loc = max(face_locs, key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))
                face_encs = face_recognition.face_encodings(rgb, [largest_loc])
                
                if face_encs:
                    matches = face_recognition.compare_faces(known_encs, face_encs[0], 0.6)
                    if True in matches:
                        match_name = known_names[matches.index(True)]
                        if match_name != current_user:
                            current_user = match_name
                            total_blinks = 0
                            dry_eye_events = 0 
                            session_start = time.time()
                            last_boost_time = 0 
                            bpm_history = []
                            fatigue_alert = False 
                            csv_path = os.path.join(DATA_DIR, f"{current_user}.csv")
                            
                            intensity_path = os.path.join(DATA_DIR, f"{current_user}_intensity.txt")
                            if os.path.exists(intensity_path):
                                with open(intensity_path, 'r') as f:
                                    try: humidifier_intensity = int(f.read().strip())
                                    except: humidifier_intensity = 50
                            else: humidifier_intensity = 50
                            
                            if not os.path.exists(csv_path):
                                session_id = 1
                                with open(csv_path, 'w', newline='') as f:
                                    csv.writer(f).writerow(["Session_ID", "Timestamp", "Total_Blinks", "BPM", "Status"])
                            else:
                                with open(csv_path, 'r') as f:
                                    reader = list(csv.reader(f))
                                    session_id = int(reader[-1][0]) + 1 if len(reader) > 1 else 1
                    else:
                        if current_user not in ["Scanning...", "Unregistered"]:
                            print("\n[WARN] Unregistered person detected! Pausing active session.")
                            current_user = "Unregistered"
                            session_start = None  
                            csv_path = None       
                            fatigue_alert = False
                            last_boost_time = 0
        else:
            if current_user == "Scanning...":
                print("[WARN] No face detected. Searching...", end="\r")
        last_id_check = time.time()

    # 2. BLINK & FATIGUE TRACKING
    if main_rect is not None and current_user not in ["Scanning...", "Unregistered"]:
        shape = predictor(gray, main_rect)
        shape = face_utils.shape_to_np(shape)
        ear = (eye_aspect_ratio(shape[36:42]) + eye_aspect_ratio(shape[42:48])) / 2.0

        if ear < 0.23: 
            COUNTER += 1
            if COUNTER >= FATIGUE_FRAMES: fatigue_alert = True
        else:
            if COUNTER >= 2:
                total_blinks += 1
                if csv_path and session_start:
                    bpm_history.append(bpm)
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([session_id, datetime.now().strftime("%H:%M:%S"), total_blinks, round(bpm, 1), status_text])
            COUNTER = 0
            fatigue_alert = False 

    # 3. BPM & STATUS MATH
    status_color = (0, 0, 0)
    if session_start and current_user not in ["Scanning...", "Unregistered"]:
        elapsed = time.time() - session_start
        dur_min = elapsed / 60
        bpm = total_blinks / dur_min if dur_min > 0.02 else 20.0 

        if bpm >= 20: 
            status_text, status_color = "Healthy", (0, 255, 0)
            last_boost_time = 0 
        elif 10 <= bpm < 20: 
            status_text, status_color = "Moderate", (0, 255, 255)
            last_boost_time = 0 
        else: 
            status_text, status_color = "Severe", (0, 0, 255)
            
            # --- AUTO BOOST LOGIC ---
            if elapsed > 20 and (time.time() - last_boost_time > 20.0):
                dry_eye_events += 1 
                
                humidifier_intensity = min(100, humidifier_intensity + 10)
                with open(os.path.join(DATA_DIR, f"{current_user}_intensity.txt"), 'w') as f:
                    f.write(str(humidifier_intensity))
                
                duration_seconds = int(humidifier_intensity / 10)
                if arduino is not None:
                    try: arduino.write(f"{duration_seconds}\n".encode())
                    except: pass
                        
                print(f"\n[AUTO] Status is Severe! Spraying for {duration_seconds}s.")
                last_boost_time = time.time() 

        # --- FIREBASE LIVE UPDATE FOR HTML DASHBOARD (Every 3 seconds) ---
        if firebase_enabled and (time.time() - last_firebase_update > 3.0):
            try:
                daily_ref = db.reference("dryeyescem")
                daily_ref.update({
                    "blinkCount": total_blinks,
                    "dryeyeDetection": dry_eye_events,
                    "humidifireStatus": 1 if status_text == "Severe" else 0,
                    "temperature": current_temp,     # NEW: Sending real sensor data
                    "humidity": current_humidity     # NEW: Sending real sensor data
                })
                last_firebase_update = time.time()
            except Exception as e:
                pass 
    else:
        bpm = 0.0
        status_text = "Paused" if current_user == "Unregistered" else "Scanning"
        status_color = (0, 0, 255) if current_user == "Unregistered" else (0, 0, 0)

    # 4. COMPACT UI & MANUAL CONTROLS
    FONT = cv2.FONT_HERSHEY_COMPLEX
    H_SIZE, V_SIZE, STEP = 0.55, 0.50, 25
    BLACK = (0, 0, 0)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):
        cv2.destroyAllWindows() 
        print("\n" + "="*40)
        new_name = input("[NEW REGISTRATION] Enter Name: ").strip()
        if new_name:
            print(f"Capturing face for {new_name}... Please look at the camera.")
            fresh_frame = vs.read()
            if fresh_frame is not None:
                rgb_fresh = cv2.cvtColor(fresh_frame, cv2.COLOR_BGR2RGB)
                face_locs_fresh = face_recognition.face_locations(rgb_fresh, model="hog")
                if face_locs_fresh:
                    largest_loc = max(face_locs_fresh, key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))
                    new_enc = face_recognition.face_encodings(rgb_fresh, [largest_loc])[0]
                    matches = face_recognition.compare_faces(known_encs, new_enc, 0.6)
                    if True in matches:
                        existing_name = known_names[matches.index(True)]
                        print(f"\n[WARN] Face is already registered under the name: '{existing_name}'!")
                        print("[INFO] Ending new registration. Resuming active session...")
                    else:
                        np.save(os.path.join(PROFILES_DIR, f"{new_name}.npy"), new_enc)
                        print(f"[SUCCESS] {new_name} registered! Resuming system...")
                        known_encs, known_names = load_known_faces() 
                else:
                    print("[ERROR] Could not see a face. Registration failed. Resuming...")
        print("="*40 + "\n")

    elif key == ord('h') or key == ord('l'):
        if key == ord('h'): humidifier_intensity = min(100, humidifier_intensity + 10)
        elif key == ord('l'): humidifier_intensity = max(0, humidifier_intensity - 10)
            
        if current_user not in ["Scanning...", "Unregistered"]:
            with open(os.path.join(DATA_DIR, f"{current_user}_intensity.txt"), 'w') as f:
                f.write(str(humidifier_intensity))

    # --- MAIN UI DRAWING ---
    cv2.putText(frame, "USER:", (15, 30), FONT, H_SIZE, BLACK, 2)
    cv2.putText(frame, str(current_user), (85, 30), FONT, V_SIZE, BLACK, 1)
    cv2.putText(frame, "BPM:", (15, 30 + STEP), FONT, H_SIZE, BLACK, 2)
    cv2.putText(frame, f"{bpm:.1f}", (70, 30 + STEP), FONT, V_SIZE, BLACK, 1)
    cv2.putText(frame, "LEVEL:", (15, 30 + (STEP * 2)), FONT, H_SIZE, BLACK, 2)
    cv2.putText(frame, f"{humidifier_intensity}%", (95, 30 + (STEP * 2)), FONT, V_SIZE, BLACK, 1)
    cv2.putText(frame, "STATUS:", (15, 30 + (STEP * 3)), FONT, H_SIZE, BLACK, 2)
    cv2.circle(frame, (115, 30 + (STEP * 3) - 5), 6, status_color, -1)
    cv2.putText(frame, status_text, (135, 30 + (STEP * 3)), FONT, V_SIZE, status_color, 2 if status_text != "Scanning" else 1)

    if current_user not in ["Scanning...", "Unregistered"] and fatigue_alert:
        cv2.putText(frame, "ALERT:", (15, 30 + (STEP * 4)), FONT, H_SIZE, BLACK, 2)
        cv2.circle(frame, (115, 30 + (STEP * 4) - 5), 6, (0, 0, 255), -1) 
        cv2.putText(frame, "FATIGUE", (135, 30 + (STEP * 4)), FONT, V_SIZE, (0, 0, 255), 2)

    cv2.imshow("Dry Eye IoT System", frame)
    if key == ord('q'): break

cv2.destroyAllWindows(); vs.stop()
if arduino is not None: arduino.close()

# --- 5. GRAPH DISPLAY AFTER QUIT ---
if bpm_history and current_user not in ["Scanning...", "Unregistered"]:
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