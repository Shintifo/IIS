import os
import argparse
import sqlite3

BASE_DIR = r'datasets'


def create_DB(db_name):
	conn = sqlite3.connect(f"databases/{db_name}")
	c = conn.cursor()

	c.execute('''
	    CREATE TABLE IF NOT EXISTS images (
	        id INTEGER PRIMARY KEY AUTOINCREMENT,
	        file_name TEXT,
	        image BLOB
	    )
	''')

	conn.commit()
	conn.close()


def insert_image(image_path, db_name):
	conn = sqlite3.connect(db_name)
	connection = conn.cursor()

	with open(image_path, 'rb') as f:
		image_blob = f.read()

	image_name = image_path.split("/")[-1]

	connection.execute('''
        INSERT INTO images (file_name, image_name)
        VALUES (?, ?)
    ''', (image_name, image_blob))

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
		insert_image(img_path, db_name)
