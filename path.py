import cv2
import numpy as np
import tensorflow as tf
import pygame
from twilio.rest import Client
import time
import face_recognition
import requests
from io import BytesIO
import os
from PIL import Image


# Initialize pygame mixer for sound
pygame.mixer.init()

# Twilio cre   # Recipient's phone number

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

# Function to download and process the image from a GitHub repository
def download_face_image(image_url):
    response = requests.get(image_url)
    img_data = BytesIO(response.content)
    img = Image.open(img_data)
    img = np.array(img)
    img_rgb = img[:, :, ::-1]  # Convert to RGB (from BGR)
    return img_rgb

# Function to load multiple face images from a GitHub repository
def load_known_faces_from_github(github_repo_url):
    known_faces = []
    known_names = []
    
    # Correct URLs for images
    image_urls = [
        "https://raw.githubusercontent.com/Durgalgorithm/Sneha_project/main/adarsh.jpg",
        "https://raw.githubusercontent.com/Durgalgorithm/Sneha_project/main/sneha.jpg",
        # Add more URLs for each person’s image
    ]

    for url in image_urls:
        try:
            image_name = os.path.basename(url).split('.')[0]  # Extract name from filename
            known_image = download_face_image(url)
            face_encodings = face_recognition.face_encodings(known_image)
            if face_encodings:  # Ensure at least one face is detected
                known_encoding = face_encodings[0]  # Get encoding of the known face
                known_faces.append(known_encoding)
                known_names.append(image_name)  # Use the filename (without extension) as the person's name
        except Exception as e:
            print(f"Error loading face from {url}: {e}")
    
    return known_faces, known_names

# Load known faces from the GitHub repository
github_repo_url = "https://github.com/Durgalgorithm/Sneha_project"
known_faces, known_names = load_known_faces_from_github(github_repo_url)

alert_sent = False  # Prevent multiple alerts for the same detection
last_alert_time = 0  # Track time of last alert

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert the image to RGB (OpenCV uses BGR)
    rgb_frame = frame[:, :, ::-1]

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        
        # Display the name on the frame
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Preprocess the frame for object detection
    input_frame = preprocess_frame(frame)

    # Make predictions for object detection
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

    # Show the frame with detected face and prediction
    cv2.imshow("Object and Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows

