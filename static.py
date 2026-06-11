import cv2
import numpy as np
import tensorflow as tf
import pygame
from twilio.rest import Client
import time

# Initialize pygame mixer for sound
pygame.mixer.init()

# Twilio cre
# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load the labels
with open('C:\\Users\\Aadarsh\\Desktop\\testing\\knife.txt', 'r') as f:
    labels = f.read().splitlines()

# Function to preprocess the input frame for prediction
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model input size
    normalized_frame = resized_frame / 255.0       # Normalize the frame
    return np.expand_dims(normalized_frame, axis=0) # Add batch dimension

# Function to play alarm sound
def play_alarm():
    pygame.mixer.Sound(r'C:\Users\Aadarsh\Desktop\sound\alarm.WAV').play()

# Function to send SMS alert with error handling
def send_sms_alert():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 ALERT: Dangerous object detected! Immediate action required.",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("✅ SMS Sent Successfully:", message.sid)
    except Exception as e:
        print("❌ SMS Sending Failed:", e)

# Function to make an alert phone call with error handling
def make_call():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml="<Response><Say>Alert! Dangerous object detected. Take immediate action!</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print("✅ Call Initiated Successfully:", call.sid)
    except Exception as e:
        print("❌ Call Initiation Failed:", e)

# Initialize the camera
camera = cv2.VideoCapture(0)
alert_sent = False  # Prevent multiple alerts for the same detection
last_alert_time = 0  # Track time of last alert

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(input_frame)
    predicted_label = labels[np.argmax(predictions)]  # Get the label with the highest confidence
    confidence = np.max(predictions)  # Get the confidence score

    # Display prediction on the frame
    cv2.putText(frame, f"{predicted_label} ({confidence*100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Trigger alarm, SMS, and call if "gun" is detected with > 80% confidence
    if predicted_label.lower() == "gun" and confidence > 0.80:
        play_alarm()
        
        # Only send alert if 10 seconds have passed since last alert
        current_time = time.time()
        if not alert_sent or current_time - last_alert_time > 10:
            send_sms_alert()
            make_call()
            alert_sent = True  # Set flag to avoid spamming alerts
            last_alert_time = current_time  # Update last alert time

    # Reset alert flag if gun is no longer detected
    if predicted_label.lower() != "gun":
        alert_sent = False

    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
