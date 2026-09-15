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

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
clf = joblib.load(os.path.join(base_dir, "logreg_model.joblib"))

LABELS = {0: "REAL", 1: "AI-GENERATED"}

def predict_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()

    prob = clf.predict_proba(embedding)[0]
    fake_prob = prob[1]

    if 0.45 <= fake_prob <= 0.55:
        final_label = "INCONCLUSIVE"
        confidence = float(fake_prob if fake_prob > 0.5 else 1.0 - fake_prob)
    else:
        pred = 1 if fake_prob > 0.55 else 0
        final_label = LABELS[pred]
        confidence = float(prob[pred])

    return {
        "label": final_label,
        "confidence": round(confidence, 4)
    }