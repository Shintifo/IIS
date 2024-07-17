import sqlite3
import numpy as np
import cv2  # OpenCV for image handling

conn = sqlite3.connect('images_embeddings.db')
c = conn.cursor()

c.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB,
        embedding BLOB
    )
''')

conn.commit()
conn.close()




def insert_image_embedding(image_path, embedding):
    conn = sqlite3.connect('images_embeddings.db')
    c = conn.cursor()

    # Read image
    with open(image_path, 'rb') as f:
        image_blob = f.read()

    # Convert embedding to binary
    embedding_blob = np.array(embedding).tobytes()

    # Insert the image and embedding into the table
    c.execute('''
        INSERT INTO images (image, embedding)
        VALUES (?, ?)
    ''', (image_blob, embedding_blob))

    conn.commit()
    conn.close()




insert_image_embedding('TMF.jpg', [0.1, 0.2, 0.3, 1,1,2,312,312,312,3,123,123,123,124,124,12])