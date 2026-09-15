from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.predict import predict_image
from model.gradcam import generate_gradcam

app = FastAPI(title="SignalScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "SignalScope API is running"}

from model.metadata import extract_metadata

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    heatmap = generate_gradcam(image_bytes)
    meta = extract_metadata(image_bytes)

    result["heatmap"] = heatmap
    result["metadata"] = meta
    return result
from model.multimodal import check_image_text_consistency
from fastapi import Form

@app.post("/predict_multimodal")
async def predict_multimodal(file: UploadFile = File(...), caption: str = Form(...)):
    image_bytes = await file.read()
    verdict = predict_image(image_bytes)
    heatmap = generate_gradcam(image_bytes)
    text_check = check_image_text_consistency(image_bytes, caption)

    return {**verdict, "heatmap": heatmap, "text_consistency": text_check}
from model.metadata import extract_metadata

