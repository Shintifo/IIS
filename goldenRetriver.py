#use this to retrieve single file from db
import sqlite3
import numpy as np
import cv2
import sys

def retrieve_image_embedding(image_id):
    conn = sqlite3.connect('images_embeddings.db')
    c = conn.cursor()

    # Retrieve the image and embedding
    c.execute('''
        SELECT image, embedding FROM images WHERE id=?
    ''', (image_id,))

    row = c.fetchone()

    # Convert binary data back to original formats
    image_blob = row[0]
    embedding_blob = row[1]
    image = np.frombuffer(image_blob, dtype=np.uint8)
    embedding = np.frombuffer(embedding_blob, dtype=np.float32)

    conn.close()

    # Decode image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image, embedding


if len(sys.argv) != 2:
    print("Usage: python script.py <image_id>")
    sys.exit(1)

image_id = int(sys.argv[1])
image, embedding = retrieve_image_embedding(image_id)
cv2.imshow("Retrieved Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
