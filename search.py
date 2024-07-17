# search in db
import sys
import os
import sqlite3

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from run import main

DATASET = "custom"
DATABASE_NAME = f"{DATASET}.db"


def retrieve_image_embedding(image_id):
	conn = sqlite3.connect(DATABASE_NAME)
	cur = conn.cursor()

	# Retrieve the image and embedding
	cur.execute('''
        SELECT image, embedding FROM images WHERE id=?
    ''', (image_id,))

	row = cur.fetchone()

	# Convert binary data back to original formats
	image_blob = row[0]
	embedding_blob = row[1]
	image = np.frombuffer(image_blob, dtype=np.uint8)
	embedding = np.frombuffer(embedding_blob, dtype=np.float32)

	conn.close()

	# Decode image
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image, embedding


def tinker_display(images, resize_width=600):
	# Create a Tkinter window
	root = tk.Tk()
	root.title("Retrieved Images")

	# Resize images and determine the largest image size
	resized_images = []
	max_width = 0
	max_height = 0

	for img in images:
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(img_rgb)

		# Resize image while maintaining aspect ratio
		aspect_ratio = img_pil.width / img_pil.height
		new_width = resize_width
		new_height = int(resize_width / aspect_ratio)
		img_resized = img_pil.resize((new_width, new_height))

		resized_images.append(img_resized)

		# Update max dimensions
		if new_width > max_width:
			max_width = new_width
		if new_height > max_height:
			max_height = new_height

	# Set initial window size based on the largest image
	initial_width = max_width * 3 + 40  # 3 columns + padding
	initial_height = max_height + 20  # Add some padding
	root.geometry(f"{initial_width}x{initial_height}")

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

	num_columns = 3

	for i, img_resized in enumerate(resized_images):
		img_tk = ImageTk.PhotoImage(img_resized)
		label = tk.Label(scrollable_frame, image=img_tk)
		label.image = img_tk  # Keep a reference to avoid garbage collection
		row = i // num_columns
		column = i % num_columns
		label.grid(row=row, column=column, padx=5, pady=5)

	root.mainloop()


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Not correct input!")
		print("Usage: python script.py <image_id>")
		sys.exit(1)

	base_dir = r'.\datasets'
	images_dir = os.path.join(base_dir, DATASET)
	query_image = str(sys.argv[1])

	inds = main(DATASET, query_image)

	images = []
	for i in inds:
		print("Retrieving image", i + 1)
		image, embedding = retrieve_image_embedding(int(i + 1))
		images.append(image)

	tinker_display(images)