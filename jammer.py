import cv2
import numpy as np
import tensorflow as tf
import pygame
from twilio.rest import Client
import time
import face_recognition
import requests
from io import BytesIO
from PIL import Image
import os
from ultralytics import YOLO

# ================== INITIAL SETUP ==================
pygame.mixer.init()

# Twilio cre
  

# Alarm sound path
ALARM_SOUND_PATH = r"C:\Users\Aadarsh\Desktop\sound\Alarmss.mp3"

# Load Keras model
MODEL_PATH = "model.h5"
LABEL_PATH = r"C:\Users\Aadarsh\Desktop\testing\knife.txt"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Custom model loaded successfully")
except Exception as e:
    print("❌ Model load error:", e)
    exit()

# Load YOLOv8 pretrained weights
try:
    yolo_model = YOLO("yolov8n.pt")  # small & fast model
    print("✅ YOLOv8 loaded successfully")
except Exception as e:
    print("❌ YOLO load error:", e)
    exit()

# Load labels
if not os.path.exists(LABEL_PATH):
    print(f"❌ Labels file not found at {LABEL_PATH}")
    exit()

with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

# ================== FUNCTIONS ==================
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)

def play_alarm():
    try:
        sound = pygame.mixer.Sound(ALARM_SOUND_PATH)
        sound.play()
        print("🔊 Alarm Triggered!")
    except Exception as e:
        print("❌ Alarm error:", e)

def send_sms_alert():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 ALERT: Dangerous object detected! Immediate action required.",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("✅ SMS Sent:", message.sid)
    except Exception as e:
        print("❌ SMS send error:", e)

def make_call():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml="<Response><Say>Alert! Dangerous object detected. Take immediate action!</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("✅ Call initiated:", call.sid)
    except Exception as e:
        print("❌ Call error:", e)

def download_face_image(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print("❌ Error downloading:", e)
        return None

def load_known_faces():
    known_faces = []
    known_names = []
    urls = [
        "https://raw.githubusercontent.com/Durgalgorithm/Sneha_project/main/adarsh.jpg",
    ]  
    for url in urls:
        name = os.path.basename(url).split('.')[0]
        img = download_face_image(url)
        if img is not None:
            if img.shape[0] > 800:
                scale = 800 / img.shape[0]
                img = cv2.resize(img, (int(img.shape[1]*scale), 800))
            try:
                faces = face_recognition.face_encodings(img)
                if faces:
                    known_faces.append(faces[0])
                    known_names.append(name)
                    print(f"✅ Loaded known face: {name}")
            except Exception as e:
                print(f"⚠️ Error encoding {name}:", e)
    return known_faces, known_names

# Load known faces
known_faces, known_names = load_known_faces()

# ================== CAMERA LOOP ==================
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("❌ Camera not detected!")
    exit()

alert_sent = False
last_alert_time = 0

print("🎥 Camera started... Press 'q' to exit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Frame not captured")
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- FACE RECOGNITION ----
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 128, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ---- YOLO OBJECT DETECTION ----
    yolo_results = yolo_model(frame, verbose=False)

    detected_objects = []
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            detected_objects.append(label)
            # Draw boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    danger_detected = any(obj.lower() in ["gun", "knife", "weapon"] for obj in detected_objects)

    # ---- CUSTOM MODEL PREDICTION ----
    input_frame = preprocess_frame(frame)
    preds = model.predict(input_frame, verbose=False)
    predicted_label = labels[np.argmax(preds)]
    confidence = np.max(preds)

    cv2.putText(frame, f"{predicted_label} ({confidence*100:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---- ALERT SYSTEM ----
    if danger_detected or (predicted_label.lower() in ["gun", "knife"] and confidence > 0.80):
        play_alarm()
        now = time.time()
        if not alert_sent or now - last_alert_time > 10:
            send_sms_alert()
            make_call()
            alert_sent = True
            last_alert_time = now
    else:
        alert_sent = False

    cv2.imshow("Smart Surveillance & Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("🛑 Program ended.")