import cv2
import numpy as np
import pygame
import time
import os
from twilio.rest import Client
from ultralytics import YOLO

pygame.mixer.init()
#twilio

alarm_sound_path = r"C:\Users\Aadarsh\Desktop\sound\Alarmss.mp3"

if not os.path.exists(alarm_sound_path):
    raise FileNotFoundError(f"Alarm sound not found at {alarm_sound_path}")

# Load YOLOv8 model
print("🔄 Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")  # small model = fast detection
print("✅ YOLOv8 model loaded successfully!")

# Load alarm sound
alarm_sound = pygame.mixer.Sound(alarm_sound_path)

# ================= ALERT FUNCTIONS =================
def play_alarm():
    """Play warning alarm."""
    try:
        print("🔊 Alarm Triggered!")
        alarm_sound.play()
        time.sleep(2)
    except Exception as e:
        print("❌ Error playing alarm:", e)

def send_sms_alert():
    """Send Twilio SMS alert."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 ALERT: Dangerous object detected! Immediate action required.",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("✅ SMS Sent:", message.sid)
    except Exception as e:
        print("❌ SMS Failed:", e)

def make_call():
    """Make Twilio voice call alert."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml="<Response><Say>Alert! Dangerous object detected. Take immediate action!</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("✅ Call Initiated:", call.sid)
    except Exception as e:
        print("❌ Call Failed:", e)

# ================= DETECTION LOGIC =================
def detect_objects(frame):
    """Run YOLO detection and return result frame + detection flag."""
    results = yolo_model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

    threat_detected = False

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = yolo_model.names[int(cls)]
        confidence = float(conf)

        color = (0, 255, 0)
        if label.lower() in ["knife", "gun", "firearm", "pistol", "revolver", "rifle","scissors","suitcase"]:
            color = (0, 0, 255)
            threat_detected = True

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame, threat_detected

# ================= MAIN CAMERA LOOP =================
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("❌ Camera not detected")

alert_sent = False
last_alert_time = 0
frame_count = 0

print("🎥 Camera started... Press 'q' to exit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Frame not captured")
        continue

    frame_count += 1

    # Skip frames for less lag
    if frame_count % 3 != 0:
        cv2.imshow("Weapon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Run YOLO detection
    detected_frame, threat = detect_objects(frame)

    # Trigger alerts if any threat detected
    if threat:
        play_alarm()
        current_time = time.time()
        if not alert_sent or current_time - last_alert_time > 15:
            send_sms_alert()
            make_call()
            alert_sent = True
            last_alert_time = current_time
    else:
        alert_sent = False

    # Display result
    cv2.imshow("Weapon Detection", detected_frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("🛑 Program ended.")
