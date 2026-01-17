import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the Model Architecture (Must match training correctly)
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
            nn.Linear(256, 4)  # 4 classes
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# Labels (Alphabetical order of folders)
LABELS = ['Glioma Tumor', 'Meningioma Tumor', 'Normal', 'Pituitary Tumor']

# Layout and Title
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image to detect tumor type.")

# Load Model
@st.cache_resource
def load_model():
    model = BrainTumorCNN()
    # Try loading the model file (assumes it's in the same directory)
    model_path = "cnn_model.pth"
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("⚠️ Model file 'cnn_model.pth' not found. Please place it in the same directory.")
        return None

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# File Uploader
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    
    if st.button('Predict'):
        if model:
            with st.spinner('Analyzing...'):
                # Preprocess
                img_tensor = transform(image).unsqueeze(0) # Add batch dimension
                
                # Predict
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Result
                predicted_label = LABELS[predicted_idx.item()]
                confidence_score = confidence.item() * 100
                
                st.success(f"**Prediction:** {predicted_label}")
                st.info(f"**Confidence:** {confidence_score:.2f}%")
                
                # Plot probabilities (Optional)
                st.bar_chart({label: prob.item() for label, prob in zip(LABELS, probabilities[0])})
        else:
            st.error("Model not loaded. Cannot predict.")
