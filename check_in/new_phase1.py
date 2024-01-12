import os
import cv2
import numpy as np
from deepface import DeepFace
import sqlite3
from mtcnn import MTCNN

# Connect to the SQLite database or create a new one
conn = sqlite3.connect("face_encodings_deepface.db")
cursor = conn.cursor()

# Create a table to store the face encodings and image names
cursor.execute("CREATE TABLE IF NOT EXISTS face_encodings_deepface (id INTEGER PRIMARY KEY, encoding BLOB, name TEXT, library TEXT)")

# Specify the root folder path containing the image files
root_folder = "photos"

# Create an MTCNN face detector
face_detector = MTCNN()

def extract_name_from_filename(filename):
    # Split the filename by "_" to extract the name
    parts = filename.split("_")
    if len(parts) >= 1:
        return parts[0]
    else:
        return "Unknown"

def align_face(image):
    # Detect faces in the image using MTCNN
    faces = face_detector.detect_faces(image)
    aligned_faces = []

    for face in faces:
        # Get the bounding box coordinates
        x, y, w, h = face['box']

        # Ensure the detected face is within the image boundaries
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            continue

        # Extract the face region from the image
        face_image = image[y:y+h, x:x+w]

        # Skip if the face region is too small
        if face_image.shape[0] < 10 or face_image.shape[1] < 10:
            continue

        # Resize the face image to a fixed size
        aligned_face = cv2.resize(face_image, (224, 224))
        aligned_faces.append(aligned_face)

    return aligned_faces

def process_images(folder_path, model_name):
    for root, dirs, files in os.walk(folder_path):
        # Iterate over the files in the current folder
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Align the face in the image
                aligned_faces = align_face(image)

                # Skip the image if no valid faces are detected
                if len(aligned_faces) == 0:
                    print(f"No valid faces detected in image: {file}")
                    continue

                for aligned_face in aligned_faces:
                    try:
                        # Generate the 128-dimensional face embedding using the specified model with face detection
                        embedding = DeepFace.represent(aligned_face, model_name=model_name, enforce_detection=False)[0]["embedding"]

                        # Convert the face embedding to bytes for storing in the database
                        embedding_bytes = np.array(embedding).tobytes()

                        # Extract the name from the filename
                        name = extract_name_from_filename(file)

                        # Check if the image file already exists in the database
                        cursor.execute("SELECT name FROM face_encodings_deepface WHERE name=?", (name,))
                        existing_files = cursor.fetchall()

                        if not existing_files:
                            # Insert the face encodings and image name into the database
                            cursor.execute("INSERT INTO face_encodings_deepface (encoding, name, library) VALUES (?, ?, ?)", (embedding_bytes, name, model_name))

                    except ValueError:
                        print(f"Face detection failed for image: {file}")

        # Commit the changes after processing all images in the current folder
        conn.commit()

def choose_model():
    # Prompt the user to choose a model for face recognition
    print("Available models: Facenet, VGG-Face, OpenFace")
    model_name = input("Enter the name of the model for face recognition: ")

    if model_name.lower() not in ["facenet", "vgg-face", "openface"]:
        print("Invalid model name. Using the default model: Facenet")
        model_name = "Facenet"

    return model_name

# Choose the model for face recognition
model_name = choose_model()

# Process images in the root folder and its subfolders
process_images(root_folder, model_name)

# Close the database connection
conn.close()