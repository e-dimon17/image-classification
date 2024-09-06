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


from tensorflow.keras import layers, models

def proprietary_cnn(input_shape=(224, 224, 3), num_classes=10):
    model = models.Sequential()

    # Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Compile the model
model = proprietary_cnn(num_classes=train_csv['label'].nunique())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()


from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, Input

def vgg16_model(input_shape=(224, 224, 3), num_classes=10):
    # Define the input
    inputs = Input(input_shape)

    # Define VGG16 base model
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    base_model.trainable = False  # Freeze the base model

    model = models.Model(inputs = inputs, outputs = outputs)

    return model

vgg16 = vgg16_model(num_classes=train_csv['label'].nunique())
vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg16.summary()

from tensorflow.keras.applications import ResNet50

def resnet50_model(input_shape=(224, 224, 3), num_classes=10):
    # Define input
    inputs = Input(input_shape)

    # Defien ResNet50 base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    base_model.trainable = False  # Freeze the base model

    model = models.Model(inputs = inputs, outputs = outputs)

    return model

resnet50 = resnet50_model(num_classes=train_csv['label'].nunique())
resnet50.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet50.summary()

def baseline_cnn(input_shape=(224, 224, 3), num_classes=10):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

baseline = baseline_cnn(num_classes=train_csv['label'].nunique())
baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
baseline.summary()

epochs = 10  # Adjust as needed

# Proprietary CNN
history_prop = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs)

# VGG16
history_vgg16 = vgg16.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs)

# ResNet50
history_resnet50 = resnet50.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs)

# Baseline CNN
history_baseline = baseline.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs)
