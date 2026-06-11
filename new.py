import cv2
import numpy as np
import tensorflow as tf
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()

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

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
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

    # Trigger alarm if "gun" is detected with > 80% confidence
    if predicted_label == "gun" and confidence > 0.80:
        play_alarm()

    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
