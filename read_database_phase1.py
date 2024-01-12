import sqlite3
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect("face_encodings_deepface.db")
cursor = conn.cursor()

# Retrieve all data from the face_encodings table
cursor.execute("SELECT encoding, name, library FROM face_encodings_deepface")
rows = cursor.fetchall()

# Iterate over the retrieved data
for row in rows:
    encoding_bytes = row[0]
    name = row[1]
    library = row[2]

    # Convert the encoding bytes to a NumPy array
    encoding = np.frombuffer(encoding_bytes)

    # Perform actions with the encoding and name
    print("Name:", name)
    print("Encoding:", encoding)
    print("library:", library)
    print()

# Close the database connection
conn.close()