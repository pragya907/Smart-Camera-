import cv2
import numpy as np
import pygame
import time
import os
from twilio.rest import Client
from ultralytics import YOLO
import face_recognition
import requests
from io import BytesIO
from PIL import Image
import sounddevice as sd

pygame.mixer.init()

# ------------------- TWILIO CONFIG -------------------


# ------------------- ALARM -------------------
alarm_sound_path = r"C:\Users\Aadarsh\Desktop\sound\Alarmss.mp3"
alarm_sound = pygame.mixer.Sound(alarm_sound_path)

# ------------------- YOLO -------------------
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

# Expanded harmful objects list
DANGEROUS_OBJECTS = [
    "knife", "gun", "firearm", "pistol", "revolver",
    "rifle", "ak47", "shotgun", "weapon"
]

# ------------------- ALERT FUNCTIONS -------------------
def play_alarm():
    if not pygame.mixer.get_busy():
        alarm_sound.play()

def send_sms_alert(msg="ALERT triggered!"):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=msg,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("SMS sent")
    except Exception as e:
        print("SMS error:", e)

def make_call():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.calls.create(
            twiml="<Response><Say>Emergency Alert Triggered</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("Call done")
    except Exception as e:
        print("Call error:", e)

# ------------------- FACE LOADING -------------------
def download_face_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except:
        return None

print("Loading Adarsh face...")
adarsh_url = "https://raw.githubusercontent.com/Durgalgorithm/Sneha_project/main/adarsh.jpg"
adarsh_img = download_face_image(adarsh_url)

adarsh_encoding = face_recognition.face_encodings(adarsh_img)[0]
known_faces = [adarsh_encoding]
known_names = ["Adarsh"]

# ------------------- CLAP DETECTION -------------------
clap_times = []

def detect_clap(indata, frames, time_info, status):
    global clap_times
    volume_norm = np.linalg.norm(indata) * 10

    if volume_norm > 20:  # threshold (tune if needed)
        current_time = time.time()
        clap_times.append(current_time)

        # keep only last 3 seconds
        clap_times = [t for t in clap_times if current_time - t < 3]

        if len(clap_times) >= 3:
            print("3 CLAPS DETECTED 🚨")
            play_alarm()
            send_sms_alert("Emergency via Clap Detection!")
            make_call()
            clap_times = []

# Start audio stream
stream = sd.InputStream(callback=detect_clap)
stream.start()

# ------------------- CAMERA -------------------
camera = cv2.VideoCapture(0)

FRAME_SKIP = 3
frame_count = 0

alert_sent = False
last_alert_time = 0

print("System started...")

while True:
    ret, frame = camera.read()
    if not ret:
        continue

    frame_count += 1
    display_frame = cv2.resize(frame, (800, 600))

    current_threat = False
    vip_alert = False

    if frame_count % FRAME_SKIP == 0:

        small_frame = cv2.resize(display_frame, (416, 416))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        scale_x = 800 / 416
        scale_y = 600 / 416

        # ---------------- FACE ----------------
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        faces = []

        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, enc)
            name = "Unknown"

            if True in matches:
                name = known_names[matches.index(True)]

            top = int(top * scale_y)
            right = int(right * scale_x)
            bottom = int(bottom * scale_y)
            left = int(left * scale_x)

            faces.append((name, (left, top, right, bottom)))

            color = (0,255,255) if name=="Adarsh" else (255,0,0)

            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(display_frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ---------------- YOLO ----------------
        results = yolo_model(small_frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

        persons = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = yolo_model.names[int(cls)].lower()

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            if "person" in label:
                persons.append((x1,y1,x2,y2))

            # WEAPON DETECTION
            if any(obj in label for obj in DANGEROUS_OBJECTS):
                current_threat = True
                color = (0,0,255)
            else:
                color = (0,255,0)

            cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(display_frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        # ---------------- VIP PROTECTION ----------------
        for name, (l,t,r,b) in faces:
            if name == "Adarsh":
                for (px1,py1,px2,py2) in persons:
                    dist = abs(px1 - l)

                    if dist < 100:  # proximity threshold
                        vip_alert = True
                        cv2.putText(display_frame,"VIP BREACH!",
                                    (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    # ---------------- ALERT LOGIC ----------------
    if current_threat or vip_alert:
        play_alarm()
        current_time = time.time()

        if not alert_sent or current_time - last_alert_time > 15:
            send_sms_alert("Threat or VIP breach detected!")
            make_call()
            alert_sent = True
            last_alert_time = current_time
    else:
        alert_sent = False

    cv2.imshow("Smart Security System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
stream.stop()
print("System Ended")