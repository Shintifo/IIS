# Identical Image Search System (ISS)
The Identical Image Search System (ISS) is a powerful tool designed to search for identical images using a pretrained model called SuperGlobal. It utilizes the FAISS library in Python for efficient storage and retrieval of image embeddings, and SQL for managing metadata.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Model](#model)
4. [Database](#database)
5. [License](#license)

## Installation <a name="installation"></a>

To get started with ISS, follow these steps:

1. Clone the repository: `git clone https://github.com/Shintifo/ISS.git`
2. Navigate into the project directory: `cd ISS`
3. Install the required packages: `pip install -r requirements.txt`

## Usage <a name="usage"></a>

To use ISS, you need to have your images ready in a datasets/{dataset_name} directory. 
Also, you need to create folder databases.
Firstly, you need to initialize the database with images. `python database.py dataset_name`

Then, you can use the main script to search for identical images. `python search.py dataset_name query_image`

After, it display all similar images by set threshold (threshold can be changed in `run.py`)

## Model <a name="model"></a>

ISS uses the [SuperGlobal pretrained model](https://github.com/ShihaoShao-GH/SuperGlobal) to generate image embeddings. 
This model was converted to ONNX format and executed in `run.py`.

## Database <a name="database"></a>

- **SQL**: ISS uses SQL to store and manage metadata about the images. This includes image by itself and image name.
- **FAISS**: ISS uses index based database to store vector embeddings.

## License <a name="license"></a>
ISS is released under the MIT License. See the LICENSE file for more details.
