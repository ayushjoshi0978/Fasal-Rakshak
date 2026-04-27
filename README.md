# 🌾 फसल रक्षक v2.0 — Fasal Rakshak (Offline AI)
### No API. No Internet. No Credits. Pure ML.

## Setup (2 minutes)
```bash
pip install -r requirements.txt
python main.py
```

## How it works
- Extracts color + texture features from leaf photo
- KNN classifier matches against 14 disease profiles
- Gives disease name, severity, treatment in Hindi/English

## Diseases Detected (14 classes)
| Crop | Diseases |
|------|----------|
| Tomato | Bacterial Spot, Early Blight, Late Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Wheat | Yellow Rust, Healthy |
| Rice | Leaf Scald, Healthy |
| Corn | Common Rust, Northern Leaf Blight, Healthy |

## Better Accuracy
Download PlantVillage dataset from Kaggle:
https://www.kaggle.com/datasets/emmarex/plantdisease
Then use "Train with Real Images" tab.

## Tech Stack
- Python + Tkinter (UI)
- scikit-learn KNN (ML)
- PIL/Pillow (image processing)
- numpy (feature extraction)
- Fully offline — no API needed!
