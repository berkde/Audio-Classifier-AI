import sys
import torch
import warnings
from model import Net
from Dataset import CustomAudioDataset


warnings.filterwarnings("ignore")

# -------- Parameters --------
SAMPLE_RATE = 22050
IMG_HEIGHT = 128
IMG_WIDTH = 256
NUM_CLASSES = 3
MODEL_PATH = "best_model.pth"
LABELS = ['bird', 'cat', 'dog']


def predict(file_path):
    model = Net(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    spec = CustomAudioDataset.get_spectrogram(file_path)
    input_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()

    print(f"Predicted class: {LABELS[predicted]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path/to/audio.wav")
        sys.exit(1)
    predict(sys.argv[1])
