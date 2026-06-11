import cv2
import numpy as np
import tensorflow as tf
import pygame
from twilio.rest import Client
import time
import os

# ==================== CONFIG ====================
# Twilio crede 
# Alarm sound path
ALARM_SOUND = r"C:\Users\Aadarsh\Desktop\sound\Alarmss.mp3q"

# Load model and labels
MODEL_PATH = "model.h5"
LABEL_FILE = r"C:\Users\Aadarsh\Desktop\testing\knife.txt"

# ==================== INIT ====================
pygame.mixer.init()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

if not os.path.exists(LABEL_FILE):
    print(f"❌ Label file {LABEL_FILE} not found!")
    exit()

labels = open(LABEL_FILE).read().splitlines()

# ==================== FUNCTIONS ====================
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def play_alarm():
    try:
        pygame.mixer.Sound(ALARM_SOUND).play()
        print("🔊 Alarm Triggered")
    except Exception as e:
        print("❌ Alarm error:", e)

def send_alerts():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # SMS
        client.messages.create(
            body="🚨 ALERT: Harmful object detected! Immediate action required.",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )

        # Call
        client.calls.create(
            twiml="<Response><Say>Alert! Dangerous object detected. Take immediate action!</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )

        print("✅ Alerts sent via SMS & Call")
    except Exception as e:
        print("❌ Alert sending failed:", e)

# ==================== CAMERA LOOP ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera")
    exit()

print("🎥 Camera started")

last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    input_frame = preprocess_frame(frame)
    predictions = model.predict(input_frame, verbose=0)
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    cv2.putText(frame, f"{predicted_label} ({confidence*100:.2f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if predicted_label.lower() in ["gun", "knife"] and confidence > 0.80:
        play_alarm()
        current_time = time.time()
        if current_time - last_alert_time > 10:  # Prevent spamming alerts
            send_alerts()
            last_alert_time = current_time

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Program Ended")
