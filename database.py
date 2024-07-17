# use this to append images folder to db
import os
import argparse
import sqlite3
import numpy as np

BASE_DIR = r'datasets'


def create_DB(db_name):
	conn = sqlite3.connect(db_name)
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


def insert_image_embedding(image_path, embedding, db_name):
	conn = sqlite3.connect(db_name)
	connection = conn.cursor()

	# Read image
	with open(image_path, 'rb') as f:
		image_blob = f.read()

	# Convert embedding to binary
	embedding_blob = np.array(embedding).tobytes()

	# Insert the image and embedding into the table
	connection.execute('''
        INSERT INTO images (image, embedding)
        VALUES (?, ?)
    ''', (image_blob, embedding_blob))

	conn.commit()
	conn.close()


def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str, help='Name of the dataset')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse()
	db_name = f"{args.dataset}.db"

	create_DB(db_name)

	images_dir = os.path.join(BASE_DIR, args.dataset)
	for img in os.listdir(images_dir):
		img_path = os.path.join(images_dir, img)
		insert_image_embedding(img_path, [1, 0], db_name)
