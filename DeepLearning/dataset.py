import numpy as np
import os
from tqdm import tqdm
import cv2
from zipfile import ZipFile
import urllib
from urllib import request
from config import *

def scale(data, method):   # Same for every type of data. Images, Games, Numbers
    if method == 1:
        data = (data.reshape(data.shape[0], -1).astype(np.float32) - (data.max() / 2)) / (data.max() / 2)
    elif method == 2:
        data = (data.reshape(data.shape[0], -1).astype(np.float32) / (data.max() / 2))

    return np.array(data)


def download_dataset(url, file_name, *, del_zip=True):
    if not os.path.isfile(file_name) and not os.path.isdir(
            file_name[:-4]):  # If you haven't downloaded it already then download it
        print(f"Downloading Dataset from: {url}, saving as {file_name}")
        urllib.request.urlretrieve(url, file_name)

        if file_name[-3:] == "zip":
            print('Unzipping data...')
            with ZipFile(file_name) as zip_folder:
                zip_folder.extractall(file_name[:-4])

            if del_zip:
                print('Deleting original zip file...')
                os.remove(file_name)

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in tqdm(labels):
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


def mnist_data_to_dataset(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test


def mnist_preprocessing(data_folder):
    # Create dataset from given data
    X, y, X_test, y_test = mnist_data_to_dataset(data_folder)

    # Shuffle the training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # Scale and reshape samples
    X = scale(X, scaling_method)
    X_test = scale(X_test, scaling_method)

    return X, y, X_test, y_test
