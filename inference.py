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

@app.post("/predict")
async def predict(model_type: str = Form(...), image: UploadFile = File(...)):
    print("Received model type:", model_type)
    print("Received file:", image.filename)
    # Validate the model type
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")

    # Read the image file
    img = Image.open(image.file)

    # Preprocess the image as required by your model
    img = img.resize((224, 224))  # Adjust size if needed
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Select the model based on the model_type
    model_inference = models[model_type]

    # Make prediction
    predictions = model_inference.predict(img_array)

    # Assuming your model outputs a softmax score for each class
    predicted_class = np.argmax(predictions, axis=1)

    return {"predicted_class": int(predicted_class[0])}