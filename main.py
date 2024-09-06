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

# Assuming the CSV files have a column 'filename' and 'label' or similar
train_csv['image_path'] = train_csv['filename'].apply(lambda x: os.path.join(train_dir, x))
test_csv['image_path'] = test_csv['filename'].apply(lambda x: os.path.join(test_dir, x))

# Define a function to load and preprocess images
def load_and_preprocess_image(path):
    img = load_img(path, target_size=(224, 224))  # Adjust the target size as needed
    img_array = img_to_array(img) / 255.0  # Normalize the image
    return img_array

# Apply to the DataFrame
train_csv['image'] = train_csv['image_path'].apply(load_and_preprocess_image)
test_csv['image'] = test_csv['image_path'].apply(load_and_preprocess_image)

# Define the generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% of the data will be used for validation
)

# Create a generator for training
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory=train_dir,
    x_col='filename',  # Column with filenames
    y_col='label',  # Column with labels, if available
    target_size=(224, 224),  # Adjust size as needed
    batch_size=32,
    class_mode='categorical',  # Use 'binary' or 'categorical' based on your problem
    subset='training'  # Set as training data
)

# Create a generator for validatioin
validation_generator  = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory=train_dir,
    x_col='filename',  # Column with filenames
    y_col='label',  # Column with labels, if available
    target_size=(224, 224),  # Adjust size as needed
    batch_size=32,
    class_mode='categorical',  # Use 'binary' or 'categorical' based on your problem
    subset='validation'  # Set as training data
)
