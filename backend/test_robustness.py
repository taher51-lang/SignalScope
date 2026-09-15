import torch
import clip
from PIL import Image
import joblib
import numpy as np
import os, random, io
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
clf = joblib.load("model/logreg_model.joblib")

def get_test_paths(folder, label, n_samples=500):
    random.seed(42)
    all_files = os.listdir(folder)
    sampled = random.sample(all_files, min(n_samples, len(all_files)))
    return [(os.path.join(folder, f), label) for f in sampled]

N_PER_CLASS = 500  # smaller sample since this runs multiple times (once per degradation level)
real_paths = get_test_paths("cifake_data/test/REAL", 0, N_PER_CLASS)
fake_paths = get_test_paths("cifake_data/test/FAKE", 1, N_PER_CLASS)
test_data = real_paths + fake_paths

def degrade_image(img: Image.Image, jpeg_quality=None, resize_factor=None):
    if resize_factor:
        w, h = img.size
        img = img.resize((int(w * resize_factor), int(h * resize_factor)))
        img = img.resize((w, h))  # resize back up, simulating quality loss
    if jpeg_quality:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        img = Image.open(buf)
    return img

def evaluate_condition(jpeg_quality=None, resize_factor=None):
    embeddings = []
    labels = []
    batch_images, batch_labels = [], []
    batch_size = 64

    def process_batch(imgs, labs):
        with torch.no_grad():
            image_input = torch.stack(imgs).to(device)
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy(), labs

    for path, label in test_data:
        img = Image.open(path).convert("RGB")
        img = degrade_image(img, jpeg_quality, resize_factor)
        img_tensor = preprocess(img)
        batch_images.append(img_tensor)
        batch_labels.append(label)
        if len(batch_images) == batch_size:
            feats, labs = process_batch(batch_images, batch_labels)
            embeddings.append(feats)
            labels.extend(labs)
            batch_images, batch_labels = [], []

    if batch_images:
        feats, labs = process_batch(batch_images, batch_labels)
        embeddings.append(feats)
        labels.extend(labs)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    probs = clf.predict_proba(embeddings)[:, 1]
    preds = clf.predict(embeddings)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    return auc, acc

print("=== ROBUSTNESS TO DEGRADATION ===\n")

conditions = [
    ("Original (no degradation)", {}),
    ("JPEG quality 50", {"jpeg_quality": 50}),
    ("JPEG quality 20", {"jpeg_quality": 20}),
    ("Resized 0.5x (then upscaled)", {"resize_factor": 0.5}),
    ("Resized 0.25x (then upscaled)", {"resize_factor": 0.25}),
]

results = []
for name, kwargs in conditions:
    print(f"Testing: {name}...")
    auc, acc = evaluate_condition(**kwargs)
    results.append((name, auc, acc))
    print(f"  AUC: {auc:.4f}, Accuracy: {acc:.4f}\n")

print("=== SUMMARY ===")
for name, auc, acc in results:
    print(f"{name:35s}  AUC={auc:.4f}  Acc={acc:.4f}")