import torch
import clip
from PIL import Image
import joblib
import numpy as np
import io

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load CLIP once at startup
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Load trained classifier once at startup
clf = joblib.load("model/logreg_model.joblib")

LABELS = {0: "REAL", 1: "AI-GENERATED"}

def predict_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()

    pred = clf.predict(embedding)[0]
    prob = clf.predict_proba(embedding)[0]
    confidence = float(prob[pred])

    return {
        "label": LABELS[int(pred)],
        "confidence": round(confidence, 4)
    }