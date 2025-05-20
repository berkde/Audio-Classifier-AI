import streamlit as st
import torch
import numpy as np
from model import Net
from Dataset import CustomAudioDataset
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Constants
LABELS = ['bird', 'cat', 'dog']
MODEL_PATH = "./best_model.pth"


# Load model
@st.cache_resource
def load_model():
    model = Net(num_classes=len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()


# UI
st.title("Animal Sound Classifier AI")
st.markdown("Upload a short `.wav` audio clip of a cat, dog, or bird to classify it.")

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    spec = CustomAudioDataset.get_spectrogram(uploaded_file)
    input_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        st.write("Class Probabilities:", {label: f"{p * 100:.2f}%" for label, p in zip(LABELS, probs)})
        prediction = np.argmax(probs)

    st.success(f"✅ Predicted Class: **{LABELS[prediction]}**")
