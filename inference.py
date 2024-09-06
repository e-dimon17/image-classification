from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load the saved models
models = {
    "proprietary_cnn": tf.keras.models.load_model('proprietary_cnn.h5'),
    "vgg16": tf.keras.models.load_model('vgg16_model.h5'),
    "resnet50": tf.keras.models.load_model('resnet50_model.h5'),
    "baseline_cnn": tf.keras.models.load_model('baseline_cnn.h5')
}