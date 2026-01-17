import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # 1. Download Dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("nisargpatel2466/brain-tumer-dataset")
    print("Path to dataset files:", path)

    DATA_PATH = os.path.join(path, 'archive', 'Data')
    
    # 2. Data Transformations
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    # 3. Load Dataset
    if not os.path.exists(DATA_PATH):
        print(f"Data path not found: {DATA_PATH}")
        # Fallback for structure variation if any
        DATA_PATH = os.path.join(path, 'Data') 
        if not os.path.exists(DATA_PATH):
             DATA_PATH = path # Direct path

    print(f"Loading data from: {DATA_PATH}")
    try:
        dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Split dataset: 80% train, 20% test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 4. Model Definition
    class BrainTumorCNN(nn.Module):
        def __init__(self):
            super(BrainTumorCNN, self).__init__()
            self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(128*16*16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 4)  # 4 classes: pituitary, meningioma, glioma, normal
            )

        def forward(self, x):
            x = self.conv_layer(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layer(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BrainTumorCNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    num_epochs = 4
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # 6. Evaluation
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # 7. Confusion Matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")

    # 8. Save Model
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model saved as 'cnn_model.pth'")

if __name__ == "__main__":
    main()
