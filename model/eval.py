import torch
import clip
from PIL import Image
import joblib
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP (same version used in predict.py)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Load your trained classifier
clf = joblib.load("model/logreg_model.joblib")

def get_test_paths(folder, label, n_samples=2000):
    random.seed(42)
    all_files = os.listdir(folder)
    sampled = random.sample(all_files, min(n_samples, len(all_files)))
    return [(os.path.join(folder, f), label) for f in sampled]

# 2000 per class = 4000 total, fast on M1, still statistically meaningful
N_PER_CLASS = 2000
real_paths = get_test_paths("cifake_data/test/REAL", 0, N_PER_CLASS)
fake_paths = get_test_paths("cifake_data/test/FAKE", 1, N_PER_CLASS)
test_data = real_paths + fake_paths
random.shuffle(test_data)

print(f"Evaluating on {len(test_data)} test images...")

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

for path, label in tqdm(test_data):
    img = preprocess(Image.open(path).convert("RGB"))
    batch_images.append(img)
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

# Predict
test_probs = clf.predict_proba(embeddings)[:, 1]
test_preds = clf.predict(embeddings)

# Metrics
auc = roc_auc_score(labels, test_probs)
macro_f1 = f1_score(labels, test_preds, average="macro")
cm = confusion_matrix(labels, test_preds)

print("\n=== RESULTS ON HELD-OUT CIFAKE TEST SET ===")
print(f"ROC-AUC: {auc:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(labels, test_preds, target_names=["REAL", "FAKE"]))