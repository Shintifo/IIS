#use this to append iamges folder to db
import os
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

DATASET = "custom"
base_dir = r'.\datasets'
images_dir = os.path.join(base_dir, DATASET)
for img in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img)
    insert_image_embedding(img_path, [12,12,312,1,1312,132,123,321,123])
