# Lightweight Deep Learning Model for Binary Classification of Healthy and Diseased Maize Leaves

A comparative study of lightweight deep learning architectures (MobileNetV2, InceptionV3, ResNet50, VGG16, Xception) for binary classification of healthy vs. diseased maize leaves.

## Trained Models

The trained model files (`.h5`) are hosted on Google Drive due to their size:

**[Download Trained Models](https://drive.google.com/drive/folders/1xHrPaCJ_EoAVFzPClaS6wL44Bf0fneoy?usp=sharing)**

The folder contains:
- `InceptionV3_final.h5`
- `MobileNetV2_final.h5`
- `ResNet50_final.h5`
- `VGG16_final.h5`
- `Xception_final.h5`
- `mobilenetv2_phase1_best (1).h5` and `(2).h5` — phase 1 checkpoints
- Presentation slides (`.pptx`)

## Repository Structure

```
├── Training Codes/        # Jupyter notebooks for each model
├── Results/               # Evaluation outputs per model
│   ├── *_confusion_matrix.png
│   ├── *_roc_curve.png
│   ├── *_training_history.png
│   └── *_results.json
└── Project Output/        # Final trained model storage (see Drive link above)
```

## Loading a Model

```python
from tensorflow.keras.models import load_model

model = load_model("MobileNetV2_final.h5")
```

## Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, Matplotlib, scikit-learn
