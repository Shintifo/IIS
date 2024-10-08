import os

import numpy as np
import onnxruntime
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import faiss
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 0.4
SIMILARITY_THRESHOLD = 0.3
TOP_K = 100
IN_SIZE = 500
MODEL = "WrapperONNX.onnx"


def transform_image(image_path):
	image = Image.open(image_path).convert('RGB')

	transformer = transforms.Compose([
		transforms.Resize((IN_SIZE, IN_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	image = transformer(image).unsqueeze(0).numpy()
	return image


def extract_features(images_dir):
	i_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]

	session = onnxruntime.InferenceSession(MODEL)
	input_name = session.get_inputs()[0].name
	out = []
	for img_path in tqdm(i_paths):
		out.append(session.run(None, {input_name: transform_image(img_path)})[0])

	img_feats = torch.tensor(np.vstack(out))
	X = F.normalize(img_feats, p=2, dim=1)
	return X


def qimg_embedding(qimg_path):
	q = transform_image(qimg_path)

	session = onnxruntime.InferenceSession(MODEL)
	input_name = session.get_inputs()[0].name

	q = session.run(None, {input_name: q})[0]
	q = torch.tensor(np.vstack(q))
	q = F.normalize(q, p=2, dim=1)
	return q


def save_index(X, dataset_name):
	index = faiss.IndexFlatL2(X.shape[1])
	index.add(X)
	faiss.write_index(index, f'databases/{dataset_name}.index')


def find_similar(dataset_name, query_vector, recalc):
	if not os.path.exists(os.path.join('databases', f"{dataset_name}.index")) or recalc:
		images_dir = f"{os.getcwd()}/datasets/{dataset_name}"
		X = extract_features(images_dir)
		save_index(X, dataset_name)

	index = faiss.read_index(f"databases/{dataset_name}.index")

	k = min(TOP_K, index.ntotal)

	distances, indices = index.search(query_vector, k)
	indices = np.asarray(indices).flatten()
	distances = np.asarray(distances).flatten()

	indices = indices[np.where(distances <= SIMILARITY_THRESHOLD)]

	# print(f"Indices of nearest neighbors: {indices}")
	# print(distances[:len(indices)])
	# print(distances)
	return indices


def main(dataset_name, qimg, recalc=False):
	qimg_path = os.path.join("datasets", dataset_name, qimg)
	q_vector = qimg_embedding(qimg_path)
	res = find_similar(dataset_name, q_vector, recalc)
	return res


if __name__ == "__main__":
	main('oxford5k', "all_souls_000072.jpg", True)
