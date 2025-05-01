# import modules
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "*",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the tf model
MODEL = tf.keras.models.load_model(
    "./models/potato_disease_detect_model.keras")

# this method will not work with keras v3
# MODEL = tf.keras.models.load_model(
#     "./models/v1")


@app.get("/ping")
async def ping():
    return {"status": 200, "message": "FastAPI server running, for potato disease classification."}


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    take an upload file input and predict the disease
    """
    CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
    image = read_file_as_image(await file.read())
    # each image is converted into a numpy array of (x, y, 3)
    # note user can upload image of any size; not necessarily 256x256

    image_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    # run with uvicorn
    print("FastAPI Server Started Running.....")
    uvicorn.run(app, host="0.0.0.0", port=8001)
