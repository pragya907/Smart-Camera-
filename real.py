import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from playsound import playsound

# Load the pre-trained model (replace 'model.h5' with your model file)
model = load_model('model.h5')

# Load the labels (ensure this is a text file containing the class names, one per line)
# Example file: 'labels.txt' should have class names like "knife", "gun", etc.
with open('C:\\Users\\Aadarsh\\Desktop\\testing\\knife.txt', 'r') as f:
    labels = f.read().splitlines()

# Function to preprocess input frame for model prediction
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Adjust size as per model input
    normalized_frame = resized_frame / 255.0       # Normalize the frame
    return np.expand_dims(normalized_frame, axis=0) # Add batch dimension

# Function to play an alarm sound
def play_alarm():
    playsound('C:\\Users\\Aadarsh\\Desktop\\sound\\alarm.mp3')

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

    # Trigger alarm if a "knife" or dangerous object is detected
    if predicted_label == "gun" and confidence > 0.80: # Replace with your specific label for the knife
        play_alarm()

    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
