from flask import Flask, render_template, request
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import base64
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from threading import Thread

app = Flask(__name__)
# Configuration for scope and authentication information
credentials_file = "river-engine-400013-b1d721c87331.json"
spreadsheet_id = "1qTj4g-Yy-iuscuGuyt38GCMTPKctLtZhC1s5e0qfmU8"
worksheet_name = "register_information"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
client = gspread.authorize(credentials)
sheet = client.open_by_key(spreadsheet_id).worksheet(worksheet_name)

drive_service = build('drive', 'v3', credentials=credentials)
folder_id = '1fwR8ptKuLBrypWSOe_8sfxR9ACnxoCDa'

def process_registration(name, mssv, email, phone_number, current_school_year, photo_data_urls):
    num_photos = len(photo_data_urls)
    row_data = [name, mssv, email, phone_number, current_school_year, num_photos]
    sheet.append_row(row_data)

    for i, photo_data_url in enumerate(photo_data_urls):
        photo_data = base64.b64decode(photo_data_url.split(",")[1])
        photo_filename = f"{mssv.replace(' ', '_')}_{i+1}.png"  # Change the file extension to PNG
        photo_path = os.path.join("photos", photo_filename)
        with open(photo_path, "wb") as photo_file:  # Open the file in binary mode
            photo_file.write(photo_data)

        with open(photo_path, "rb") as photo_file:  # Open the file in binary mode
            media = MediaInMemoryUpload(photo_file.read(), mimetype='image/png')  # Update the mimetype to PNG

        file_metadata = {
            'name': photo_filename,
            'parents': [folder_id]
        }
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        os.remove(photo_path)

    print("Registration successful!")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        mssv = request.form['mssv']
        email = request.form['email']
        phone_number = request.form['phone_number']
        current_school_year = request.form['current_school_year']
        photo_data_urls = request.form.getlist('photo')

        # Create a separate thread to process the registration asynchronously
        thread = Thread(target=process_registration, args=(name, mssv, email, phone_number, current_school_year, photo_data_urls))
        thread.start()

        return "Registration in progress! Please wait 10 seconds before closing this site!!!!!"

    return render_template('test_index.html')

app.static_folder = 'static'
if __name__ == '__main__':
    app.run()