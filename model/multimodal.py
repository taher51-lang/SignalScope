import torch
import clip
from PIL import Image
import io

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def check_image_text_consistency(image_bytes: bytes, caption: str):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    text_tensor = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        text_features = model.encode_text(text_tensor)

        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (img_features @ text_features.T).item()

    # CLIP cosine similarities for genuinely matching pairs are typically ~0.25-0.35
    consistency_score = round(similarity, 4)
    is_consistent = similarity > 0.20  # rough threshold, tune based on testing

    return {
        "caption": caption,
        "consistency_score": consistency_score,
        "likely_consistent": is_consistent
    }