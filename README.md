# Audio Classifier AI

[![Python](https://img.shields.io/badge/python-3.12.4-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This project classifies raw `.wav` audio recordings of animals into **cat**, **dog**, or **bird** using a custom CNN model trained on mel spectrograms.


---

##  Project Highlights

-  Custom CNN architecture with Squeeze-and-Excitation block
-  Input preprocessing using `librosa` and `scikit-image`
-  Early stopping based on validation loss
-  Achieved **98% accuracy** on test set
-  Full train/validation/test split with visualization
-  Inference-ready script included

---

##  Model Architecture

- 3 Convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPooling)
- Squeeze-and-Excitation channel attention block
- Fully connected layers with BatchNorm and Dropout

---

##  Results

- **Training Accuracy**: ~96–98%
- **Test Accuracy**: ~98%
- **Confusion Matrix** and **Classification Report** confirm consistent class-wise performance

> ℹ️ *Note: In test splits where a class is absent or not predicted, precision/recall for that class is reported as zero (via `zero_division=0`).*

---

## Folder Structure

```
├── notebook.ipynb          # Full training pipeline
├── inference.py            # CLI inference script
├── app.py                  # Streamlit web app for live demos
├── Dataset.py              # CustomAudioDataset class for loading spectrograms
├── model.py                # CNN model architecture (Net class)
├── best_model.pth          # Saved model weights
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```


---


## Setup Instructions

```bash
git clone https://github.com/berkde/Audio-Classifier-AI.git
cd Audio-Classifier-AI
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
pip install -r requirements.txt
streamlit run app.py
```


##  Run Inference

```bash
pip install -r requirements.txt
python inference.py path/to/your/audio.wav
```

You’ll get output like:
```
Predicted class: dog
```

---

##  Dependencies

Install everything via:
```bash
pip install -r requirements.txt
```

Main libraries:
- PyTorch
- Librosa
- NumPy
- scikit-learn
- scikit-image
- matplotlib

---

##  Visualization

- Training/Validation loss and accuracy over epochs
- Confusion matrix and class-wise precision/recall

---

##  Acknowledgments

Dataset: [Kaggle - Cats vs Dogs vs Birds Audio Classification](https://www.kaggle.com/datasets/warcoder/cats-vs-dogs-vs-birds-audio-classification)

---

## Author

### Berk Delibalta – [LinkedIn](https://www.linkedin.com/in/berkdelibalta/) | [GitHub](https://github.com/berkde)

---

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. The property price predictions provided by this app are generated using historical data and machine learning models trained on engineered features, and do not constitute financial, legal, or investment advice.
While every effort has been made to ensure reasonable accuracy, the model may not reflect current market conditions, zoning laws, renovation status, or other important factors. Users should consult with licensed real estate professionals before making any real-world decisions based on these predictions.
The author assumes no liability for any decisions made based on the use of this tool.

---

> This project was originally built as a personal prototype to explore regression modeling and Streamlit deployment. It is now being incrementally upgraded toward a more production-like architecture with MLOps practices. Feedback and contributions welcome.
