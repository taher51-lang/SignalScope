import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os, random
from tqdm import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- Dataset ---
class CifakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, n_per_class=5000, transform=None):
        random.seed(42)
        real_files = random.sample(os.listdir(real_dir), n_per_class)
        fake_files = random.sample(os.listdir(fake_dir), n_per_class)
        self.samples = [(os.path.join(real_dir, f), 0) for f in real_files] + \
                       [(os.path.join(fake_dir, f), 1) for f in fake_files]
        random.shuffle(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = CifakeDataset("cifake_data/train/REAL", "cifake_data/train/FAKE", n_per_class=5000, transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

# --- Model: ResNet18, replace final layer for binary classification ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# --- Train for a few epochs ---
EPOCHS = 3
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_ds)
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={acc:.4f}")

torch.save(model.state_dict(), "model/resnet18_gradcam.pt")
print("Saved model/resnet18_gradcam.pt")