import subprocess

# Define the Kaggle dataset URL
kaggle_url = "phucthaiv02/butterfly-image-classification"

# Use Kaggle API to download the file
subprocess.run(['kaggle', 'datasets', 'download', '-d', kaggle_url])

# Unzip directory
subprocess.run(['unzip', 'butterfly-image-classification.zip', '-d', 'butterfly_data'])

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

train_dir = 'butterfly_data/train'
test_dir = 'butterfly_data/test'
train_set_csv = 'butterfly_data/Training_set.csv'
test_set_csv = 'butterfly_data/Testing_set.csv'

# Load the CSV files
train_csv = pd.read_csv(train_set_csv)
test_csv = pd.read_csv(test_set_csv)