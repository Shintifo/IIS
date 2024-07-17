#search in db
import sqlite3
import numpy as np
import cv2
import sys
from run import main
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

DATASET = "custom"

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

DATASET = "custom"
base_dir = r'.\datasets'
images_dir = os.path.join(base_dir, DATASET)
query_image = str(sys.argv[1])

inds = main(DATASET, query_image)

images = []
for i in inds:
    print("Retrieving image", i + 1)
    image, embedding = retrieve_image_embedding(int(i+1))
    images.append(image)

# Create a Tkinter window
root = tk.Tk()
root.title("Retrieved Images")

# Create a canvas with a scrollbar
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(frame)
scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Convert OpenCV images to Tkinter format and display them
for img in images:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    label = tk.Label(scrollable_frame, image=img_tk)
    label.image = img_tk  # Keep a reference to avoid garbage collection
    label.pack()

root.mainloop()