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
4
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
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
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


@app.post("/upload-picture")
async def upload_file(file: UploadFile = File(...)):
    temp_file_path = os.path.join(SAVE_DIR, file.filename)
    
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    results = picture_model.predict(source=temp_file_path, conf=0.5, save=True, save_dir=SAVE_DIR)

    processed_image_path = None
    for result in results:
        annotated_image = result.plot()
        processed_image_path = os.path.join(SAVE_DIR, "processed_" + file.filename)
        
        annotated_image = Image.fromarray(annotated_image)
        annotated_image.save(processed_image_path)
        break

    if processed_image_path and os.path.exists(processed_image_path):
        return FileResponse(processed_image_path, media_type="image/png")
    else:
        return {"error": "Processed image not found. Check the model or output directory."}
    