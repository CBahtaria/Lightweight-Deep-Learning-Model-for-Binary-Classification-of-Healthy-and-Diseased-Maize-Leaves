# Lightweight Deep Learning Model for Binary Classification of Healthy and Diseased Maize Leaves

Comparative study of five CNN architectures for binary classification of healthy vs. diseased maize leaves, targeting smallholder farmers in Sub-Saharan Africa on low-specification smartphones.

**Conclusion:** MobileNetV2 achieves the highest accuracy (99.9%) at the smallest footprint (24.4 MB, 290 ms CPU) — 8× smaller than the next best model at equal or superior performance.

---

## Model Comparison

| Architecture | Accuracy | F1 | AUC-ROC | Precision | Sensitivity | Model Size | CPU Inference | Parameters |
|---|---|---|---|---|---|---|---|---|
| **MobileNetV2** | **99.90%** | **99.94%** | **1.0000** | 100.00% | 99.88% | **24.4 MB** | **290 ms** | **2.59M** |
| InceptionV3 | 99.59% | 99.75% | 1.0000 | 99.88% | 99.63% | 190.7 MB | 550 ms | 22.3M |
| VGG16 | 99.59% | 99.75% | 0.9999 | 99.63% | 99.88% | 111.8 MB | 418 ms | 14.8M |
| Xception | 99.59% | 99.75% | 0.9999 | 99.75% | 99.75% | 166.5 MB | 384 ms | 21.4M |
| ResNet50 | 97.25% | 98.36% | 0.9983 | 96.77% | 100.00% | 227.8 MB | 540 ms | 24.1M |

95% CI on MobileNetV2: [99.43%, 99.98%] — 982-sample held-out test set.

---

## Trained Models

Trained `.h5` files are hosted on Google Drive due to size:

**[Download Trained Models → Google Drive](https://drive.google.com/drive/folders/1xHrPaCJ_EoAVFzPClaS6wL44Bf0fneoy?usp=sharing)**

| File | Architecture | Size |
|---|---|---|
| `MobileNetV2_final.h5` | MobileNetV2 | 24.4 MB |
| `VGG16_final.h5` | VGG16 | 111.8 MB |
| `Xception_final.h5` | Xception | 166.5 MB |
| `InceptionV3_final.h5` | InceptionV3 | 190.7 MB |
| `ResNet50_final.h5` | ResNet50 | 227.8 MB |
| `mobilenetv2_phase1_best (1).h5` | MobileNetV2 phase-1 checkpoint | — |
| `mobilenetv2_phase1_best (2).h5` | MobileNetV2 phase-1 checkpoint | — |

---

## Repository Structure

```
├── Training Codes/
│   ├── MobilenetV2 Training.ipynb
│   ├── InceptionV3 Model Training.ipynb
│   ├── RestNet50 Training Code.ipynb
│   ├── VGG16 Training Code.ipynb
│   ├── Xception Model Training Codes.ipynb
│   └── Universal Code For Training Models.ipynb
├── Results/
│   ├── *_confusion_matrix.png
│   ├── *_roc_curve.png
│   ├── *_training_history.png
│   └── *_results.json
└── Project Output/          # trained .h5 files — see Drive link above
```

---

## Usage

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("MobileNetV2_final.h5")

img = image.load_img("leaf.jpg", target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)[0][0]
label = "Healthy" if pred > 0.5 else "Diseased"
confidence = pred if pred > 0.5 else 1 - pred
print(f"{label} ({confidence:.2%})")
```

---

## Training Setup

Two-phase transfer learning with class-weighted loss to handle the healthy/diseased imbalance (class weights: healthy 2.81, diseased 0.61):

| Phase | Layers frozen | Epochs | Best val loss (MobileNetV2) |
|---|---|---|---|
| 1 — feature extraction | All base layers | 17 | 0.0126 |
| 2 — fine-tuning | Top layers unfrozen | 20 | 0.0072 |

```
Train: 4,579 samples · Val: 982 · Test: 982
```

To reproduce — open the relevant notebook in `Training Codes/` and point `data_dir` at your dataset split into `healthy/` and `diseased/` subdirectories.

---

## Tech Stack

| Component | Detail |
|---|---|
| **Framework** | TensorFlow / Keras |
| **Base models** | ImageNet pretrained weights |
| **Training** | Two-phase transfer learning, class-weighted binary cross-entropy |
| **Evaluation** | Accuracy, F1, AUC-ROC, confusion matrix, 95% CI (Wilson score) |
| **Inference target** | CPU — low-spec Android smartphones in SSA |
| **Notebooks** | Jupyter |

---

## Results

Confusion matrices, ROC curves, and training histories for all five architectures are in `Results/`. Raw metrics per model are in `*_results.json`.
