import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import kagglehub
from PIL import Image

# ---------------- CONFIG ---------------- #

path = kagglehub.dataset_download("nisargpatel2466/brain-tumer-dataset")
DATA_PATH = os.path.join(path, "archive", "Data")
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORM ---------------- #

# Training transform with augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# For testing / new images: no augmentation
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ---------------- DATASET ---------------- #

dataset = datasets.ImageFolder(DATA_PATH, transform=train_transform)
CLASS_NAMES = dataset.classes
print("Classes:", CLASS_NAMES)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Update test dataset transform (no augmentation)
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ---------------- #

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(CLASS_NAMES))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = BrainTumorCNN().to(device)

# ---------------- LOSS & OPTIMIZER ---------------- #

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING ---------------- #

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

# ---------------- TEST ---------------- #

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ---------------- SAVE MODEL ---------------- #

torch.save({
    "model_state": model.state_dict(),
    "classes": CLASS_NAMES
}, "cnn_model.pth")

print("Model saved successfully!")

# ---------------- NEW IMAGE PREDICTION ---------------- #

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = test_transform(img)          # same preprocessing
    img = img.unsqueeze(0).to(device)  # add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)
        print(f"Predicted Class for {img_path}: {CLASS_NAMES[pred.item()]}")

# Example usage:
# predict_image("new_brain_image.jpg")
