import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import io
import base64

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Load the fine-tuned ResNet18 ---
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
resnet_model.load_state_dict(torch.load(os.path.join(base_dir, "resnet18_gradcam.pt"), map_location=device))
resnet_model.to(device)
resnet_model.eval()

# Target the last conv layer for Grad-CAM
target_layers = [resnet_model.layer4[-1]]
cam = GradCAM(model=resnet_model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_gradcam(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Run Grad-CAM (targets the "fake" class = index 1, so heatmap shows fake-driving regions)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]  # None = uses model's top prediction

    # Normalize original image to 0-1 for overlay
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Encode as base64 PNG to send over the API
    heatmap_img = Image.fromarray(visualization)
    buf = io.BytesIO()
    heatmap_img.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{heatmap_b64}"