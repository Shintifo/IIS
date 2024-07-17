import argparse
import os
import time

import numpy as np
import onnxruntime
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import faiss
from tqdm import tqdm


def transform_image(image_path):
	image = Image.open(image_path)

	transformer = transforms.Compose([
		transforms.Resize((500, 500)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	image = transformer(image).unsqueeze(0).numpy()
	return image


def extract_features(images_dir):
	i_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]

	imgs = [transform_image(img_path) for img_path in i_paths]

	session = onnxruntime.InferenceSession(f"WrapperONNX.onnx")
	out = []
	for img in tqdm(imgs):
		out.append(session.run(None, {"l_tensor_": img})[0])

	img_feats = torch.tensor(np.vstack(out))
	X = F.normalize(img_feats, p=2, dim=1)
	return X


def qimg_embedding(qimg_path):
	Q = transform_image(qimg_path)

	session = onnxruntime.InferenceSession(f"WrapperONNX.onnx")
	q = session.run(None, {"l_tensor_": Q})[0]

	q = torch.tensor(np.vstack(q))
	q = F.normalize(q, p=2, dim=1)
	return q


def save_index(X):
	index = faiss.IndexFlatL2(2048)
	index.add(X)
	faiss.write_index(index, 'index_file.index')


def fais(X, query_embedding):
	save_index(X)
	index = faiss.read_index("index_file.index")

	k = min(100, X.shape[0])

	distances, indices = index.search(query_embedding, k)

	indices = indices[0]  # TODO change to numpy
	distances = distances[0]
	indices = indices[np.where(distances <= 0.25)]
	print(f"Indices of nearest neighbors: {indices}")

	for i in indices:  # TODO change to numpy
		print(distances[i])


def main(dataset_name, qimg):
	images_dir = f"{os.getcwd()}/datasets/{dataset_name}"
	qimg_path = os.path.join(images_dir, qimg)

	s = time.time()
	X = extract_features(images_dir)
	print(time.time() - s)

	Q = qimg_embedding(qimg_path)
	fais(X, Q)


if __name__ == "__main__":
	main('custom', "A1.jpg")
