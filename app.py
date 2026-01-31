import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="wide"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
}
.title-text {
    font-size: 48px;
    font-weight: 800;
    color: #38bdf8;
}
.subtitle {
    font-size: 20px;
    color: #cbd5f5;
}
.card {
    background: #020617;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}
.result-box {
    background: #020617;
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown('<div class="title-text">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered MRI Classification System</div><br>', unsafe_allow_html=True)

# ------------------ CNN MODEL ------------------
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
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
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("cnn_model.pth", map_location=device)

    model = BrainTumorCNN(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint["classes"], device

model, tumor_classes, device = load_model()

# ------------------ Image Transform ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ------------------ Layout ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose MRI Image",
        type=["jpg", "jpeg", "png"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preview")
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, width=350)
    else:
        st.info("Upload MRI image to preview")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Prediction Section ------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

if uploaded_file:
    if st.button("🔍 Analyze Tumor"):
        with st.spinner("Analyzing MRI..."):
            img = Image.open(uploaded_file).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                predicted_index = torch.argmax(output, dim=1).item()
                tumor_name = tumor_classes[predicted_index]

        st.markdown(
            f'<div class="result-box">Detected Tumor: {tumor_name}</div>',
            unsafe_allow_html=True
        )
else:
    st.warning("Please upload an MRI image.")

st.markdown('</div>', unsafe_allow_html=True)
