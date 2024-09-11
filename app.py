import uvicorn
import os 
from flask_cors import CORS
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import ORJSONResponse
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from os import environ

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# Load variables from the .env file
load_dotenv()

# Load environment variables
 
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
REGION_NAME = os.getenv("REGION_NAME")


app = FastAPI()
# Access the variables
 


from tensorflow.keras.models import load_model # Load the saved model 
# model = load_model("model4new.keras")

from PIL import Image
from io import BytesIO

@app.post('/predict', response_class=ORJSONResponse) 
async def predict_image(file: UploadFile = File(...)): 
    # Read the uploaded image file 
    # contents = await file.read() 
    # # # Open the image 
    # img = Image.open(BytesIO(contents)) 
    # # # Resize and preprocess the image for the model 
    # img = img.resize((256, 256)) 
    # # # Resize to the model's input  size
    # img_array = image.img_to_array(img) 
    # img_array = np.expand_dims(img_array, axis=0) # Add batch dimension 
    # img_array = img_array / 255.0 # Normalize pixel values # Predict on the new image 
    # predictions = model.predict(img_array) 
    # predicted_class = tf.argmax(predictions, axis=1) # Return the predicted class as a JSON response 
    # return ORJSONResponse(content={"predicted_class":  int(predicted_class.numpy()[0])})
    return "Hello"

if __name__ == '__main__':
   
    uvicorn.run(app, host="0.0.0.0", port=5001)
