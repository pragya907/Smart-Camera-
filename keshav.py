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

pygame.mixer.init()
#twilio 

alarm_sound_path = r"C:\Users\Aadarsh\Desktop\sound\Alarmss.mp3"
if not os.path.exists(alarm_sound_path):
    raise FileNotFoundError(f"Alarm sound not found at {alarm_sound_path}")
alarm_sound = pygame.mixer.Sound(alarm_sound_path)

print(" Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
print(" YOLOv8 model loaded successfully!")

def play_alarm():
    try:
        print(" Alarm Triggered!")
        if not pygame.mixer.get_busy():
            alarm_sound.play()
    except Exception as e:
        print(" Error playing alarm:", e)

def send_sms_alert():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=" ALERT: Dangerous object detected! Immediate action required.",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print(" SMS Sent:", message.sid)
    except Exception as e:
        print(" SMS Failed:", e)

def make_call():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml="<Response><Say>Alert! Dangerous object detected. Take immediate action!</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print(" Call Initiated:", call.sid)
    except Exception as e:
        print(" Call Failed:", e)

def download_face_image(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(" Error downloading face:", e)
        return None


# ================= ORIGINAL GITHUB FACE (UNCHANGED) =================
print(" Loading known faces...")
adarsh_url = "https://raw.githubusercontent.com/Durgalgorithm/Sneha_project/main/adarsh.jpg"
adarsh_img = download_face_image(adarsh_url)
if adarsh_img is None:
    raise RuntimeError(" Could not load Adarsh's image")

adarsh_encoding = face_recognition.face_encodings(adarsh_img)[0]
known_faces = [adarsh_encoding]
known_names = ["Adarsh"]
print(" Adarsh face loaded successfully!")
# ===================================================================


# ================= NEW LOCAL FACE (Aadarsh Pandey) ==================
aadarsh_local_path = r"C:\Users\Aadarsh\Desktop\MAJOR\AadarshPandey.jpeg"

if not os.path.exists(aadarsh_local_path):
    raise FileNotFoundError("Aadarsh Pandey image not found!")

aadarsh_img_local = face_recognition.load_image_file(aadarsh_local_path)
aadarsh_encoding_local = face_recognition.face_encodings(aadarsh_img_local)

if len(aadarsh_encoding_local) == 0:
    raise RuntimeError("No face found in Aadarsh Pandey image!")

aadarsh_encoding_local = aadarsh_encoding_local[0]

print(" Aadarsh Pandey face loaded successfully (local)!")
# ===================================================================


# ================= NEW FUNCTION ==================
def check_aadarsh_pandey(face_encoding):
    global last_alert_time

    distances = face_recognition.face_distance([aadarsh_encoding_local], face_encoding)
    match_distance = distances[0]

    if match_distance < 0.65:  # STRICT threshold
        print(f" Aadarsh Pandey detected (distance={match_distance:.2f})")

        current_time = time.time()

        # Only alert once every 20 seconds
        if current_time - last_alert_time > 20:
            play_alarm()
            send_sms_alert()
            make_call()
            last_alert_time = current_time
# =================================================


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError(" Camera not detected")

cached_face_locations = []
cached_face_names = []
cached_yolo_detections = [] 
current_threat = False 

alert_sent = False
last_alert_time = 0
frame_count = 0

FRAME_SKIP = 3
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
PROCESS_WIDTH = 416
PROCESS_HEIGHT = 416

print(" Camera started... Press 'q' to exit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print(" Frame not captured")
        continue

    frame_count += 1
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)

    if frame_count % FRAME_SKIP == 0:

        cached_face_locations = []
        cached_face_names = []
        cached_yolo_detections = []
        current_threat = False 

        small_frame = cv2.resize(display_frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        scale_x = DISPLAY_WIDTH / PROCESS_WIDTH
        scale_y = DISPLAY_HEIGHT / PROCESS_HEIGHT

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog") 
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if True in matches:
                name = known_names[matches.index(True)]

            # 🚨 NEW CHECK
            check_aadarsh_pandey(face_encoding)

            top = int(top * scale_y)
            right = int(right * scale_x)
            bottom = int(bottom * scale_y)
            left = int(left * scale_x)

            cached_face_locations.append((top, right, bottom, left))
            cached_face_names.append(name)

        results = yolo_model(small_frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = yolo_model.names[int(cls)]
            confidence = float(conf)

            color = (0, 255, 0)
            if confidence > 0.6 and label.lower() in ["knife", "gun", "firearm", "pistol", "revolver", "rifle"]:
                color = (0, 0, 255)
                current_threat = True 

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            label_text = f"{label} ({confidence*100:.1f}%)"
            cached_yolo_detections.append(((x1, y1, x2, y2), label_text, color))

        if not current_threat:
            alert_sent = False 

    for (top, right, bottom, left), name in zip(cached_face_locations, cached_face_names):
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 128, 255), 2)
        cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    for (box, label, color) in cached_yolo_detections:
        (x1, y1, x2, y2) = box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if current_threat:
        play_alarm()
        current_time = time.time()

        if not alert_sent or current_time - last_alert_time > 15:
            send_sms_alert()
            make_call()
            alert_sent = True
            last_alert_time = current_time

    cv2.imshow("Smart Security System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print(" Program ended.")