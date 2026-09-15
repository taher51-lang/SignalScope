---
title: SignalScope
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# SignalScope
**Telling Real From Synthetic in the Age of Generative Media**

SignalScope is a web application that classifies an uploaded image as **real** or **AI-generated**, returns a confidence score, and — as the headline bonus feature — shows a visual explanation of which regions of the image drove that verdict.

Built for SIH 2026 (Internal Hackathon), L. J. Institute of Engineering and Technology — Problem Statement 2.

---

## 1. Modules Built

| Module | Status | Description |
|---|---|---|
| **Core Task** | ✅ Built | Real-vs-AI-generated image classification with confidence score, evaluated with ROC-AUC, macro-F1, and confusion matrix on a held-out test set. |
| **Bonus A — Faithful Explanation** | ✅ Built | Grad-CAM heatmap overlay showing which image regions most influenced the "AI-generated" verdict, toggled inline in the UI. |
| Bonus B — Generator Attribution | ❌ Not built | — |
| **Bonus C — Robustness to Degradation** | ✅ Built | Standalone evaluation script that re-tests the classifier under JPEG re-compression (quality 50, 20) and resizing (0.5x, 0.25x), reporting AUC/accuracy at each degradation level. See Section 5. |
| **Bonus D — Provenance & Metadata** | ✅ Built | Fast byte-level C2PA / Content Credentials detection and EXIF metadata extraction (camera make/model) returned with every prediction and shown in the UI. Combined with the visual verdict. |
| **Bonus E — Multimodal (image + text)** | ✅ Built | Optional caption input; CLIP text encoder computes image–caption cosine similarity as a consistency signal, served via a separate `/predict_multimodal` endpoint and shown in the UI. |
| Bonus F — Real-Time / Deployable | ✅ Partially | Deployed as a live drag-and-drop web app (see Live Demo below). |
| Bonus G — Active Defence Analysis | ❌ Not built | — |

---

## 2. Live Demo

- **Demo video (3–5 min):** [Watch on Loom](https://www.loom.com/share/009d337e982a40a58d078529c70f842d)
- **Deployment:** This project is designed to be run locally for the hackathon evaluation to avoid cloud hardware limits on the PyTorch/CLIP models. An ad-hoc public URL is generated via Localtunnel during the live pitch.

---

## 3. Architecture Overview

```
Image upload (React frontend)
        │
        ▼
FastAPI backend  ──►  CLIP ViT-B/32 (frozen) ──► LogisticRegression ──► verdict + confidence
        │
        ├──────────►  ResNet18 (fine-tuned) ──► Grad-CAM ──► explanation heatmap
        │
        ├──────────►  PIL EXIF reader ──► camera/metadata signal
        │
        └──────────►  (optional, if caption provided)
                       CLIP text encoder ──► image–caption cosine similarity
        │
        ▼
JSON response { label, confidence, heatmap, metadata, text_consistency? } ──► rendered in UI
```

**Verdict model:** Images are embedded using OpenAI's pretrained CLIP (ViT-B/32) image encoder — no fine-tuning of CLIP itself. A `scikit-learn` `LogisticRegression` classifier is trained on top of these 512-dimensional embeddings to predict real vs. AI-generated. The model uses a **calibrated confidence threshold**: if the confidence falls between 0.45 and 0.55, it returns an "Inconclusive / Not Sure" label to avoid over-claiming on borderline images. This transfer-learning approach lets a lightweight classifier leverage CLIP's broad, general-purpose visual representations, which generalize better to generators unseen during training.

**Explanation model:** A separate ResNet18 (ImageNet-pretrained, fine-tuned on the same data) drives Grad-CAM, since Grad-CAM requires the spatial feature maps of a convolutional network — CLIP's ViT backbone doesn't expose these directly. This model is used only to generate the heatmap, not the final verdict.

**Metadata check (Module D):** Fast byte-level scanning detects cryptographic C2PA / Content Credentials signatures (JUMBF). Additionally, EXIF tags (camera make/model) are read directly from the uploaded file with PIL. Presence of consistent camera metadata is weak supporting evidence of a real photograph; absence is not proof of AI generation. Reported as supporting context alongside the visual verdict, never as a standalone claim.

**Multimodal consistency (Module E):** When a caption is supplied, CLIP's text encoder embeds it into the same vector space as the image, and cosine similarity between the two embeddings is reported as a consistency score. This flags cases where an image and its accompanying claim/caption don't semantically match.

**Serving:** FastAPI exposes two endpoints — `POST /predict` (verdict + heatmap + metadata) and `POST /predict_multimodal` (all of the above, plus caption consistency when a caption is supplied).

**Frontend:** React + Vite + Tailwind CSS, with a drag-and-drop upload zone, an optional caption field, a result card with a confidence bar, a toggle between the original image and the Grad-CAM overlay, and metadata/consistency readouts.

---

## 4. Datasets Used

| Dataset | Role | License / Citation |
|---|---|---|
| **CIFAKE** (Bird & Lotfi, 2024) | Primary training + evaluation set. 100,000 train / 20,000 test images; real images from CIFAR-10, fake images generated with Stable Diffusion v1.4. | Bird, J.J. and Lotfi, A., 2024. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access. Real images: Krizhevsky, A., & Hinton, G. (2009). *Learning multiple layers of features from tiny images.* [Kaggle link](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |

> **Note on generator diversity:** CIFAKE's fake images all come from a single generator (Stable Diffusion v1.4) at low resolution (32×32, upscaled from CIFAR-10). This is a known limitation — see Section 8, Limitations.

---

## 5. Reported Metrics

### Validation split (held out from CIFAKE training pool, never trained on)
| Metric | Score |
|---|---|
| ROC-AUC | 0.9809 |
| Macro-F1 | 0.9300 |
| Confusion Matrix | `[[1398, 102], [108, 1392]]` |

### CIFAKE held-out test set (fully separate from training/validation)
| Metric | Score |
|---|---|
| ROC-AUC | 0.9775 |
| Macro-F1 | 0.9230 |
| Accuracy (threshold 0.5) | 92.30% |
| False Positive Rate (FPR) | 7.35% |
| Confusion Matrix | `[[1853, 147], [161, 1839]]` |
| Sample size | 4,000 images (2,000 real, 2,000 fake), stratified random sample |

The close agreement between validation (0.9809) and test (0.9775) AUC indicates the model is not overfitting to the training distribution.

### Informal generalization check (out-of-distribution)
A single full-resolution, photorealistic AI-generated image (outside CIFAKE's style and resolution entirely) was correctly classified as AI-generated with 88.96% confidence. A separate test image scored close to the decision boundary (~52%), suggesting the model's confidence is less reliable on images that diverge significantly from CIFAKE's 32×32, Stable-Diffusion-v1.4 training distribution. This is discussed further in Limitations.

### Bonus module results

**Module C — Robustness to degradation:** evaluated by re-running the held-out test classifier under JPEG re-compression (quality 50, 20) and resizing (0.5x, 0.25x, then upscaled back). Reproduce with:
```bash
python3 backend/test_robustness.py
```
| Degradation | AUC | Accuracy |
|---|---|---|
| Original (no degradation) | 0.9791 | 0.9230 |
| JPEG quality 50 | 0.9640 | 0.9040 |
| JPEG quality 20 | 0.9111 | 0.8100 |
| Resized 0.5× (then upscaled) | 0.9179 | 0.7330 |
| Resized 0.25× (then upscaled) | 0.7564 | 0.6880 |

The model retains strong discriminative ability (AUC > 0.91) under moderate degradation (JPEG 50, resize 0.5×). Performance degrades gracefully; even at severe 0.25× downscaling, AUC remains above 0.75.

**Module D — Metadata:** tested on both a real, unprocessed photo and a screenshot/downloaded image. Genuine EXIF data (camera make/model) was found on directly-transferred camera photos; images that had passed through messaging apps, screenshots, or social platforms — including some genuinely real photos — had no EXIF data, confirming that EXIF absence alone is not a reliable fake indicator (see Limitations).

**Module E — Multimodal consistency:** tested with a real photo and a caption that matched its actual content, returning a consistency score of 0.2202 (flagged as consistent under our threshold of 0.20). CLIP cosine similarities for genuinely matching image-caption pairs typically fall in the ~0.25–0.35 range; scores noticeably below this threshold indicate a caption that doesn't semantically match the image.

---

## 6. Setup & Run Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- ~500MB disk space for model weights + dependencies

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
The API will be live at `http://localhost:8000`. Test it with:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. Make sure the backend is running first.

### Reproducing the reported metrics
```bash
cd backend
python3 evaluate_test_set.py
```
Requires the CIFAKE dataset downloaded locally (see `cifake_data/` — not included in this repo due to size; download from the [Kaggle link](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) above and place under `backend/cifake_data/`).

A judge should be able to go from clone to a working prediction on a new image in under 10 minutes using the Backend steps above.

---

## 7. Model & Training Details

| Field | Detail |
|---|---|
| Task | Binary classification: real vs. AI-generated |
| Backbone (verdict) | CLIP ViT-B/32 (frozen, OpenAI pretrained) |
| Classifier (verdict) | Logistic Regression (`scikit-learn`, `C=1.0`, `max_iter=1000`) |
| Backbone (explanation) | ResNet18 (ImageNet-pretrained), fine-tuned for 3 epochs |
| Training data | 20,000 CIFAKE images (10,000 real, 10,000 fake), balanced random sample |
| Split | 85% train / 15% validation, stratified |
| Explanation method | Grad-CAM on ResNet18's final convolutional layer (`layer4[-1]`) |

---

## 8. Limitations

- **Resolution/style mismatch:** CIFAKE images are 32×32 pixels (upscaled from CIFAR-10), while real-world images the app will encounter are typically full-resolution. The model performs well within CIFAKE's own distribution (0.9775 test AUC) but shows reduced confidence on some out-of-distribution, full-resolution images.
- **Single-generator training data:** All fake training images come from Stable Diffusion v1.4. Newer or different generator families (Midjourney, DALL-E 3, newer SDXL variants) were not part of training, which is precisely the generalization challenge this hackathon is designed to test — and is expected to be the primary source of any AUC drop on the organizers' unseen-generator split.
- **Explanation model is separate from the verdict model:** The Grad-CAM heatmap is produced by an independently trained ResNet18, not the CLIP+LogReg model that generates the actual verdict. The heatmap should be read as an illustrative indicator of visually suspicious regions, not a mathematically exact attribution of the LogReg model's decision.
- **EXIF metadata is weak, easily-absent evidence:** Testing showed that genuinely real photos frequently have no EXIF data once they've passed through a messaging app, screenshot, or social platform — all of which strip metadata by default. Module D's output should be read as supporting context, never as a standalone real/fake signal.
- **Multimodal consistency threshold is approximate:** The 0.20 consistency threshold used in Module E is a rough heuristic based on general CLIP similarity behavior, not tuned on a labeled matched/mismatched caption dataset specific to this project. It should be treated as indicative, not calibrated.
- **Explanation, metadata, and multimodal modules run independently:** The heatmap, EXIF check, and caption consistency score are three separate signals returned alongside the verdict; the app does not currently combine them into a single unified confidence score.
- **Small informal generalization sample:** The out-of-distribution check in Section 5 is based on a small number of manually tested images, not a rigorous benchmark.

---

## 9. Originality Declaration

Per Section 8 of the challenge brief, the following third-party code, models, and datasets were used:

- **CLIP (ViT-B/32)** — pretrained model by OpenAI, used unmodified as a frozen feature extractor. [github.com/openai/CLIP](https://github.com/openai/CLIP)
- **`grad-cam` (PyPI package `grad-cam`, imported as `pytorch_grad_cam`)** — open-source library used to generate Grad-CAM visualizations. No modification to the library itself.
- **CIFAKE dataset** — Bird & Lotfi (2024), used as described in Section 4.
- **ResNet18 (ImageNet weights)** — `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, fine-tuned by us on CIFAKE for the explanation module.
- **FastAPI, scikit-learn, PyTorch, React, Vite, Tailwind CSS** — standard open-source frameworks/libraries, used as infrastructure, not as pre-built solutions to the core task.

No public real-vs-fake detection notebook was copied. The CLIP+LogisticRegression architecture and ResNet18+Grad-CAM explanation pipeline were built from scratch for this submission, informed by the general (published, not copied) approach of using frozen vision-language model features with a lightweight classifier for cross-generator generalization.

---

## 10. Repository Structure

```
/README.md              ← this file
/requirements.txt       ← Python dependencies
/backend
  main.py                ← FastAPI app, /predict and /predict_multimodal endpoints
/model
  predict.py              ← CLIP + LogisticRegression inference (verdict)
  gradcam.py               ← ResNet18 + Grad-CAM explanation generation (Module A)
  metadata.py               ← EXIF metadata extraction (Module D)
  multimodal.py              ← CLIP-based image-caption consistency (Module E)
  evaluate_test_set.py        ← held-out test set evaluation script
  train_gradcam_model.py       ← ResNet18 fine-tuning script
  test_robustness.py            ← degradation robustness evaluation (Module C)
  logreg_model.joblib     ← trained verdict classifier
  resnet18_gradcam.pt      ← fine-tuned explanation model weights
/frontend
  /src                     ← React application
  package.json
/report
  model_report.md          ← one-page model report (see separate file)
```