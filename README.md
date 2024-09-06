# Butterfly Image Classification

This project demonstrates the use of various convolutional neural network (CNN) architectures for butterfly image classification. It includes model training, evaluation, and deployment using FastAPI.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Training](#model-training)
- [API Usage](#api-usage)

## Introduction

The project performs the following tasks:
- Downloads the butterfly image classification dataset from Kaggle.
- Preprocesses the images for training.
- Trains multiple CNN architectures (Proprietary CNN, VGG16, ResNet50, and Baseline CNN).
- Evaluates model performance.
- Serves the trained models through a FastAPI application to predict the class of uploaded butterfly images.

## Requirements

Ensure you have the following dependencies installed:

```bash
tensorflow
fastapi
uvicorn
pandas
numpy
Pillow
```

Install the required packages:

```bash
pip install tensorflow fastapi uvicorn pandas numpy Pillow
```

## Dataset

The dataset used in this project is the Butterfly Image Classification dataset, which you can download from Kaggle. Ensure you have the Kaggle API set up to download the dataset.

## How to Run

1. **Download the dataset**:
   - The dataset will be automatically downloaded when you run `main.py`.

2. **Train the models**:
   - Run `main.py` to perform data preprocessing and model training.

   ```bash
   python main.py
   ```

   The trained models will be saved as `proprietary_cnn.h5`, `vgg16_model.h5`, `resnet50_model.h5`, and `baseline_cnn.h5`.

3. **Start the FastAPI server**:
   - Run `inference.py` to start the FastAPI server.

   ```bash
   python inference.py
   ```

   The server will run and provide an endpoint for accessing the API.

4. **Test the API**:
   You can use tools like Postman or `curl` to send POST requests to the API.

   Example `curl` request:

   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" -F "model_type=proprietary_cnn" -F "image=@path/to/your/image.jpg"
   ```

   The response will return the predicted class for the uploaded image.

## Model Training

The `main.py` script performs the following:

1. Downloads the Butterfly Image Classification dataset from Kaggle.
2. Loads and preprocesses the images.
3. Defines and trains multiple CNN architectures:
   - Proprietary CNN
   - VGG16
   - ResNet50
   - Baseline CNN
4. Evaluates the models and prints the test accuracy.
5. Saves the trained models to disk.

## API Usage

The `inference.py` script uses FastAPI to serve predictions based on the trained models. You can send data in the following format:

```plaintext
POST /predict
Form Data:
- model_type: "proprietary_cnn" (or "vgg16", "resnet50", "baseline_cnn")
- image: [upload your image file]
```

The response will provide the predicted class for the uploaded image.