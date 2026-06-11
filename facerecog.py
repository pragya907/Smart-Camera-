import os
import requests
from io import BytesIO
import face_recognition
from PIL import Image
import numpy as np

# Function to download and process an image from a URL
def download_face_image(image_url):
    try:
        response = requests.get(image_url)
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img = np.array(img)
        img_rgb = img[:, :, ::-1]  # Convert to RGB (from BGR)
        return img_rgb
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None

# Function to load known faces from a list of image URLs
def load_known_faces_from_github(image_urls):
    known_faces = []
    known_names = []

    for url in image_urls:
        try:
            # Extracting the name from the URL (e.g., 'adarsh.jpg' -> 'adarsh')
            image_name = os.path.basename(url).split('.')[0]  # Use filename as name
            known_image = download_face_image(url)

            if known_image is not None:
                # Detect faces in the image and get encodings
                face_encodings = face_recognition.face_encodings(known_image)
                if face_encodings:
                    known_encoding = face_encodings[0]  # Get the first encoding found
                    known_faces.append(known_encoding)
                    known_names.append(image_name)  # Store the name
                else:
                    print(f"No faces detected in the image: {image_name}")
            else:
                print(f"Skipping {image_name} due to download failure.")
        except Exception as e:
            print(f"Error loading face from {url}: {e}")
    
    return known_faces, known_names

# Example list of face images from GitHub (you can add more URLs)
image_urls = [
    "https://raw.githubusercontent.com/Durgalgorithm/Sneha_project/main/adarsh.jpg",
    "https://raw.githubusercontent.com/Durgalgorithm/Sneha_project/main/sneha.jpg",
    # Add more URLs for additional images
]

# Load known faces
known_faces, known_names = load_known_faces_from_github(image_urls)

# Print out the loaded names for verification
print("Loaded Known Faces:")
for name in known_names:
    print(f"- {name}")
