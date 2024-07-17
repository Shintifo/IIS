import argparse
import os
import math
import time

import numpy as np
import onnxruntime
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

import faiss


def show_all_similar(folder_path, similar_images, query_image):
	n = len(similar_images) + 1
	rows = 1
	cols = n
	if n > 2:
		rows = math.ceil(math.sqrt(n))
		cols = math.ceil(n / rows)

	# Create a figure and a set of subplots
	fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

	# Ensure axs is always a 2D array
	if n == 1:
		axs = np.array([[axs]])
	elif n == 2:
		axs = np.array([axs])

	# Display query image
	img = mpimg.imread(folder_path + "/" + query_image)
	axs[0, 0].imshow(img)
	axs[0, 0].set_title(f'Query Image\n{query_image}')
	axs[0, 0].axis('off')

	# Display similar images
	for i, item in enumerate(similar_images.items()):
		image, prob = item
		img = mpimg.imread(folder_path + "/" + image)
		row = (i + 1) // cols
		col = (i + 1) % cols
		axs[row, col].imshow(img)
		axs[row, col].set_title(f'Similar Image {i + 1}:\n{image}\n Probability: {prob}')
		axs[row, col].axis('off')

	# Hide any empty subplots
	for i in range(n, rows * cols):
		row = (i + 0) // cols
		col = (i + 0) % cols
		axs[row, col].axis('off')

	# Adjust the layout
	plt.tight_layout()
	plt.show()


def find_similar(qimg, X, images_dir):
	imgs = np.array([file for file in os.listdir(images_dir)])
	qimg_id = np.where(imgs == qimg)[0]
	Q = X[qimg_id]

	# Calculate matrix of similarity
	sim = torch.matmul(X, Q.T).numpy()

	# Collect names with probs
	prob_dict = {img: prob[0] for img, prob in zip(imgs, sim)}
	prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

	print("All images:")
	for k, v in prob_dict.items():
		print(f"{k}\t\t{v}")

	# Remove not similar
	threshold = 0.75
	prob_dict = {k: math.ceil(v * 10 ** 5) / 10 ** 5 for k, v in prob_dict.items() if v >= threshold and k != qimg}

	if len(prob_dict) == 0:
		print("No Similar Images!")
		return

	print("Similar images:")
	for k, v in prob_dict.items():
		print(f"{k}\t\t{v}")

	show_all_similar(images_dir, prob_dict, qimg)
	return


def extract_features(images_dir):
	imgs = [Image.open(os.path.join(images_dir, img)) for img in os.listdir(images_dir)]

	t = transforms.Compose([
		transforms.Resize((500, 500)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	imgs = [t(img).unsqueeze(0).numpy() for img in imgs]

	session = onnxruntime.InferenceSession(f"WrapperONNX.onnx")
	out = [session.run(None, {"l_tensor_": img})[0] for img in imgs]
	img_feats = torch.tensor(np.vstack(out))
	X = F.normalize(img_feats, p=2, dim=1)
	return X


def qimg_embedding(qimg, X, images_dir):
	imgs = np.array([file for file in os.listdir(images_dir)])
	qimg_id = np.where(imgs == qimg)[0]
	Q = X[qimg_id]
	return Q

def save_index(X):
	index = faiss.IndexFlatL2(2048)
	index.add(X)
	faiss.write_index(index, 'index_file.index')


def fais(X, query_embedding):
	save_index(X)
	index = faiss.read_index("index_file.index")

	k = X.shape[0]

	distances, indices = index.search(query_embedding, k)
	indices = indices[np.where(distances <= 0.25)]
	print(f"Indices of nearest neighbors: {indices}")
	print(distances)


def main(dataset_name):
	images_dir = f"{os.getcwd()}/datasets/{dataset_name}"
	s = time.time()
	X = extract_features(images_dir)
	print(time.time() - s)
	Q = qimg_embedding("A1.jpg", X, images_dir)
	fais(X, Q)


# find_similar(qimg, X, images_dir=os.path.join(os.getcwd(), "datasets/custom/jpg"))


if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description='Find Similar Images')
	# parser.add_argument('qimg', type=str, help='Query Image')
	# args = parser.parse_args()
	main('custom')
