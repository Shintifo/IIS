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


def extract_features(images_dir):
	imgs = [Image.open(os.path.join(images_dir, img)) for img in os.listdir(images_dir)]

	t = transforms.Compose([
		transforms.Resize((500, 500)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	imgs = [t(img).unsqueeze(0).numpy() for img in imgs]
	session = onnxruntime.InferenceSession(f"WrapperONNX.onnx")

	out = []
	for img in tqdm(imgs):
		out.append(session.run(None, {"l_tensor_": img})[0])
	# out = [session.run(None, {"l_tensor_": img})[0] for img in imgs]
	img_feats = torch.tensor(np.vstack(out))
	X = F.normalize(img_feats, p=2, dim=1)
	return X


def qimg_embedding(qimg, X, images_dir):
	imgs = np.array([file for file in os.listdir(images_dir)])
	qimg_id = np.where(imgs == qimg)[0]
	Q = X[qimg_id]
	return Q


def count_qimg(path_qimg):
	image = Image.open(path_qimg)

	t = transforms.Compose([
		transforms.Resize((500, 500)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	q = t(image).unsqueeze(0).numpy()

	session = onnxruntime.InferenceSession(f"WrapperONNX.onnx")
	q = session.run(None, {"l_tensor_": q})[0]
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

	indices = indices[0]
	distances = distances[0]
	indices = indices[np.where(distances <= 0.25)]
	print(f"Indices of nearest neighbors: {indices}")
	for i in indices:
		print(distances[i])


def main(dataset_name, qimg):
	images_dir = f"{os.getcwd()}/datasets/{dataset_name}"
	s = time.time()
	X = extract_features(images_dir)
	print(time.time() - s)
	Q = qimg_embedding(qimg, X, images_dir)
	fais(X, Q)


if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description='Find Similar Images')
	# parser.add_argument('qimg', type=str, help='Query Image')
	# args = parser.parse_args()
	# fais(None, count_qimg("datasets/roxford5k/all_souls_000002.jpg"))
	main('roxford5k', "all_souls_000002.jpg")
