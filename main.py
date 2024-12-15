import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile ,HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
from mimetypes import guess_type
from PIL import Image
import xgboost as xgb
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore, storage
from uuid import uuid4
import io
import base64
from dotenv import load_dotenv
import json

load_dotenv()

FIREBASE_ADMIN_SDK = os.environ.get('FIREBASE_ADMIN_SDK')

#Firebase implement
cred = credentials.Certificate(FIREBASE_ADMIN_SDK)
firebase_admin.initialize_app(cred)
db = firestore.client()
image_collection = db.collection("images")

class ItemBase(BaseModel) :
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

class ItemCreated(ItemBase):
    pass

class ItemResponse(ItemBase):
    id: int
    class Config:
        from_attributes = True #ORM mode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

picture_model = YOLO("model-ai/best.pt")
loaded_model = xgb.Booster()
loaded_model.load_model("model-ai/xgb_model.json")

SAVE_DIR = "../images"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.get("/")
def hello_world():
    return {"msg": "ok!"}

@app.post("/upload-data")
def upload_data(item : ItemCreated ):
    for field, value in item.dict().items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise HTTPException(status_code=400, detail=f"Invalid value for {field}")

    input_data = pd.DataFrame([item.dict()])
    dmatrix = xgb.DMatrix(input_data)

    prediction = loaded_model.predict(dmatrix)

    return {"input": item.dict(), "prediction": prediction.tolist()}

@app.post("/upload-picture-tofirestore")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be an image (jpeg/png/jpg)")

    file_bytes = await file.read()
    image = Image.open(io.BytesIO(file_bytes))
    image = image.convert("RGB")

    results = picture_model.predict(source=image, conf=0.5)

    if results:
        annotated_image = results[0].plot()

        annotated_image_pil = Image.fromarray(annotated_image)

        processed_image_blob = io.BytesIO()
        if file.content_type == "image/png":
            annotated_image_pil.save(processed_image_blob, format="PNG")
        elif file.content_type in ["image/jpeg", "image/jpg"]:
            annotated_image_pil.save(processed_image_blob, format="JPEG")

        processed_image_blob.seek(0)

        processed_image_base64 = base64.b64encode(processed_image_blob.read()).decode('utf-8')

        metadata = {
            "original_filename": file.filename,
            "processed_image_base64": processed_image_base64,
            "content_type": file.content_type,
        }

        try:
            doc_ref = image_collection.add(metadata)
            if isinstance(doc_ref, tuple):
                doc_id = doc_ref[0]
            else:
                doc_id = doc_ref.id

            return {"message": "Image processed and saved successfully", "file_id": doc_id}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving to Firestore: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Image processing failed.")

@app.get("/get-image-data/{file_name}")
async def get_image_data(file_name: str):
    try:
        image_collection = db.collection('images')

        query = image_collection.where('original_filename', '==', file_name)
        results = query.stream()

        images = [doc.to_dict() for doc in results]

        if not images:
            raise HTTPException(status_code=404, detail="No image found with the given filename")

        return {"images": images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading from Firestore: {str(e)}")