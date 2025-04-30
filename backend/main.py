# import modules
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# load the tf model
MODEL = tf.keras.models.load_model(
    "./models/potato_disease_detect_model.keras")


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
    image = read_file_as_image(await file.read())
    # each image is converted into a numpy array of (x, y, 3)
    # note user can upload image of any size; not necessarily 256x256

    image_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(image_batch)

    return

if __name__ == "__main__":
    # run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
