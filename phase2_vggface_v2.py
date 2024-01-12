import cv2
from deepface import DeepFace
import sqlite3
import numpy as np
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from deepface.commons import functions

credentials_file = "river-engine-400013-b1d721c87331.json"
spreadsheet_id = "1qTj4g-Yy-iuscuGuyt38GCMTPKctLtZhC1s5e0qfmU8"
worksheet_name = "database"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
client = gspread.authorize(credentials)
sheet = client.open_by_key(spreadsheet_id).worksheet(worksheet_name)

# Connect to the SQLite database
conn = sqlite3.connect("face_encodings_deepface_vgg_face.db")
cursor = conn.cursor()
face_names = []
sent_names = set()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

def calculate_similarity(embedding1, embedding2):
    # Convert the embeddings to NumPy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Calculate the cosine similarity between the embeddings
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

def calculate_confidence(similarity):
    # Calculate the confidence score based on the cosine similarity
    confidence = similarity
    return confidence

def recognize_face(face_embedding):
    # Retrieve all face encodings and names from the database
    cursor.execute("SELECT encoding, name FROM face_encodings_deepface")
    rows = cursor.fetchall()
    max_similarity = -1
    recognized_name = "Unknown"
    # Iterate over the retrieved rows
    for row in rows:
        # Retrieve the stored embedding and name
        stored_embedding = np.frombuffer(row[0], dtype=np.float64)
        stored_name = row[1]
        # Calculate the cosine similarity between the face embedding from the camera and the stored embedding
        similarity = calculate_similarity(face_embedding, stored_embedding)
        # Update the recognized name if the similarity is greater than the maximum similarity so far
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = stored_name
    # Calculate the confidence score based on the similarity
    confidence = calculate_confidence(max_similarity)
    # Adjust the threshold for considering a face as "Unknown"
    if confidence < 0.6:  # Adjust the threshold as needed
        recognized_name = "Unknown"
    return recognized_name, confidence

# Open the default camera (change index if you have multiple cameras)
camera_default = cv2.VideoCapture(0)

# Open the RTSP camera
cam_path = 'rtsp://admin:Dlhcmut@k20@192.168.1.13:554/'
camera_rtsp = cv2.VideoCapture(cam_path)

# Load the VGGFace model for face recognition
model = DeepFace.build_model("VGG-Face")

# Initialize the face detector from MTCNN
face_detector = MTCNN()

while True:
    # Read frames from the cameras
    ret_default, frame_default = camera_default.read()
    ret_rtsp, frame_rtsp = camera_rtsp.read()

    # Convert frames to RGB format
    frame_default_rgb = cv2.cvtColor(frame_default, cv2.COLOR_BGR2RGB)
    frame_rtsp_rgb = cv2.cvtColor(frame_rtsp, cv2.COLOR_BGR2RGB)

    # Detect faces in the frames using MTCNN
    faces_default = face_detector.detect_faces(frame_default_rgb)
    faces_rtsp = face_detector.detect_faces(frame_rtsp_rgb)

    # Iterate over the detected faces for the default camera
    for face_default in faces_default:
        x, y, w, h = face_default['box']
        face_region_default = frame_default_rgb[y:y+h, x:x+w]
        preprocessed_face_default = functions.preprocess_face(face_region_default, target_size=(224, 224), enforce_detection=False)
        embedding_default = model.predict(np.expand_dims(preprocessed_face_default, axis=0))[0]

        if len(embedding_default) > 0:
            recognized_name_default, confidence_default = recognize_face(embedding_default)

            if recognized_name_default:
                cv2.rectangle(frame_default, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text_default = f"{recognized_name_default}: {confidence_default:.2f}"
                cv2.putText(frame_default, text_default, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Iterate over the detected faces for the RTSP camera
    for face_rtsp in faces_rtsp:
        x, y, w, h = face_rtsp['box']
        face_region_rtsp = frame_rtsp_rgb[y:y+h, x:x+w]
        preprocessed_face_rtsp = functions.preprocess_face(face_region_rtsp, target_size=(224, 224), enforce_detection=False)
        embedding_rtsp = model.predict(np.expand_dims(preprocessed_face_rtsp, axis=0))[0]

        if len(embedding_rtsp) > 0:
            recognized_name_rtsp, confidence_rtsp = recognize_face(embedding_rtsp)

            if recognized_name_rtsp:
                cv2.rectangle(frame_rtsp, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text_rtsp = f"{recognized_name_rtsp}: {confidence_rtsp:.2f}"
                cv2.putText(frame_rtsp, text_rtsp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frames from both cameras
    resize_frame_default=cv2.resize(frame_default, (640, 480))
    resize_frame_rtsp=cv2.resize(frame_rtsp, (640, 480))
    cv2.imshow("Face Recognition - Default Camera", resize_frame_default)
    cv2.imshow("Face Recognition - RTSP Camera", resize_frame_rtsp)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close the database connection
camera_default.release()
camera_rtsp.release()
cv2.destroyAllWindows()
conn.close()
