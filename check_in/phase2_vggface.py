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

def update_checkbox(sheet, row, column, value):
    cell = sheet.cell(row, column)
    cell.value = value
    sheet.update_cells([cell])


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

# Hàng và cột bắt đầu trong Google Sheets
start_row_sheets = 6
start_column_sheets = 1

# Cập nhật một ô checkbox trong Google Sheets
def update_checkbox(sheet, row, column, value):
    cell = sheet.cell(row, column)
    cell.value = value
    sheet.update_cells([cell])

# Hàm để lấy vị trí ô trống tiếp theo trong cột
def get_next_empty_row(sheet, column):
    values = sheet.col_values(column)
    next_empty_row = start_row_sheets + len(values)
    return next_empty_row

# Hàm để push danh sách recognized_name trước khi nhận diện
def push_names_to_sheet(sheet, names):
    # Xóa dữ liệu cũ trong cột NGÀY TỔ CHỨC, THỜI ĐIỂM CHECK IN và đặt checkbox về False
    cell_range = sheet.range(start_row_sheets, start_column_sheets + 1, start_row_sheets + len(names) - 1, start_column_sheets + 2)
    for cell in cell_range:
        cell.value = ""
    checkbox_range = sheet.range(start_row_sheets, start_column_sheets + 3, start_row_sheets + len(names) - 1, start_column_sheets + 3)
    for checkbox_cell in checkbox_range:
        checkbox_cell.value = False

    # Ghi danh sách names vào cột STT và MSSV
    for i, name in enumerate(names):
        sheet.update_cell(start_row_sheets + i, start_column_sheets, i + 1)  # Cập nhật STT
        sheet.update_cell(start_row_sheets + i, start_column_sheets + 1, name)  # Cập nhật MSSV

# Lấy danh sách recognized_name từ SQLite và push vào Google Sheets
cursor.execute("SELECT name FROM face_encodings_deepface")
rows = cursor.fetchall()
recognized_names = [row[0] for row in rows]
push_names_to_sheet(sheet, recognized_names)
# Open the default camera (change index if you have multiple cameras)
# cam_path = 'rtsp://admin:Dlhcmut@k20@192.168.1.12:554/'
# camera = cv2.VideoCapture(cam_path) 
camera = cv2.VideoCapture(0)
# Load the VGGFace model for face recognition
model = DeepFace.build_model("VGG-Face")

# Initialize the face detector from MTCNN
face_detector = MTCNN()

while True:
    # Read the camera frame
    ret, frame = camera.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using MTCNN
    faces = face_detector.detect_faces(frame_rgb)

    # Iterate over the detected faces
    for face in faces:
        # Extract the bounding box coordinates
        x, y, w, h = face['box']

        # Align the face region using landmarks (if needed)
        # ...

        # Crop the face region from the frame
        face_region = frame_rgb[y:y+h, x:x+w]

        # Preprocess the face region for VGGFace model
        preprocessed_face = functions.preprocess_face(face_region, target_size=(224, 224), enforce_detection=False)
        # Generate the 128-dimensional face embedding using the VGGFace model
        embedding = model.predict(np.expand_dims(preprocessed_face, axis=0))[0]

        # Check if any embeddings were generated
        if len(embedding) > 0:
            face_embedding = embedding

            # Recognize the face by comparing the embedding with the stored encodings
            recognized_name, confidence = recognize_face(face_embedding)

            if recognized_name:

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display the recognized name and confidence above the face rectangle
                text = f"{recognized_name}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),2)
                if recognized_name not in sent_names and recognized_name != "Unknown":
                        # Ghi thời gian hiện tại
                        current_time = now.strftime("%H:%M:%S")

                        # Kiểm tra xem recognized_name đã có trong sheet chưa
                        cell = sheet.find(recognized_name, in_column=start_column_sheets + 1)

                        # Nếu recognized_name đã có trong sheet
                        if cell:
                            # Lấy hàng của recognized_name
                            row_of_recognized_name = cell.row

                            # Cập nhật NGÀY TỔ CHỨC và THỜI ĐIỂM CHECK IN
                            sheet.update_cell(row_of_recognized_name, start_column_sheets + 2, current_date)  # Cập nhật NGÀY TỔ CHỨC
                            sheet.update_cell(row_of_recognized_name, start_column_sheets + 3, current_time)  # Cập nhật THỜI ĐIỂM CHECK IN

                            # Cập nhật ô checkbox (CÓ MẶT)
                            checkbox_column = start_column_sheets + 4  # Giả sử ô checkbox nằm ở cột THỜI ĐIỂM CHECK IN + 2
                            update_checkbox(sheet, row_of_recognized_name, checkbox_column, True)  # True để tích ô checkbox

                            # Cập nhật tập hợp sent_names
                            sent_names.add(recognized_name)
                        else:
                            # Nếu recognized_name chưa có trong sheet, thêm vào cuối
                            next_empty_row = get_next_empty_row(sheet, start_column_sheets)
                            sheet.update_cell(next_empty_row, start_column_sheets, next_empty_row - start_row_sheets + 1)  # Cập nhật STT
                            sheet.update_cell(next_empty_row, start_column_sheets + 1, recognized_name)  # Cập nhật MSSV
                            sheet.update_cell(next_empty_row, start_column_sheets + 2, current_date)  # Cập nhật NGÀY TỔ CHỨC
                            sheet.update_cell(next_empty_row, start_column_sheets + 3, current_time)  # Cập nhật THỜI ĐIỂM CHECK IN

                            # Cập nhật ô checkbox (CÓ MẶT)
                            checkbox_column = start_column_sheets + 4  # Giả sử ô checkbox nằm ở cột THỜI ĐIỂM CHECK IN + 2
                            update_checkbox(sheet, next_empty_row, checkbox_column, True)  # True để tích ô checkbox

                            # Cập nhật tập hợp sent_names
                            sent_names.add(recognized_name)
    # Display the frame with recognized faces
    resized_frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Face Recognition", resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the database connection
camera.release()
cv2.destroyAllWindows()
conn.close()