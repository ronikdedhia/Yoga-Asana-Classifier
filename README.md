![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![Keras](https://img.shields.io/badge/Framework-Keras%2FTensorFlow-orange?logo=keras)
![MediaPipe](https://img.shields.io/badge/Library-MediaPipe-blue)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green)

# Yoga Asana Classifier

A real-time yoga pose (asana) recognition system that uses **MediaPipe Pose** for body landmark extraction and a **Keras dense neural network** for classification. The pipeline covers the full ML lifecycle: webcam-based data collection, model training, and live inference — all in pure Python.

## Features

- Real-time pose classification via **webcam** at inference time
- Uses **33 MediaPipe Pose landmarks** (x, y coordinates normalized to the root joint) as features
- Trains a fully-connected neural network (`Dense → Dense → Softmax`) on collected landmark data
- Visibility check ensures the model only predicts when the **full body is visible** (ankles + wrists threshold > 0.6)
- Confidence threshold (> 0.75) filters low-confidence predictions with an "Asana is either wrong or not trained" message
- Displays skeleton overlay and predicted asana name in real time
- Easily extendable — collect data for any new asana in ~80 frames

## Model Architecture

```
Input: 66 features (33 landmarks × [x, y] normalized to root)
    → Dense(128, activation='tanh')
    → Dense(64, activation='tanh')
    → Dense(num_classes, activation='softmax')

Optimizer: RMSprop
Loss: Categorical Crossentropy
Epochs: 80
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.7+ |
| Pose Estimation | MediaPipe Pose |
| Deep Learning | Keras / TensorFlow |
| Computer Vision | OpenCV |
| Data | NumPy (`.npy` files per asana) |
| Saved Model | `model.h5`, `labels.npy` |

## Getting Started

### Prerequisites

```bash
pip install mediapipe tensorflow keras opencv-python numpy
```

### Step 1 — Collect Training Data

```bash
python data_collection.py
```

Enter the asana name when prompted. Stand in front of the camera — the script captures 80 pose samples and saves them as `<asana_name>.npy`.

### Step 2 — Train the Model

```bash
python data_training.py
```

Loads all `.npy` files, shuffles data, trains the network for 80 epochs, and saves `model.h5` + `labels.npy`.

### Step 3 — Run Live Inference

```bash
python inference.py
```

Your webcam opens. Strike a yoga pose — the predicted asana name appears overlaid on screen. Press **Esc** to quit.

## File / Folder Structure

```
Yoga-Asana-Classifier/
├── data_collection.py   # Webcam-based landmark data collection
├── data_training.py     # Model training script
├── inference.py         # Real-time webcam inference
├── model.h5             # Pre-trained Keras model
└── labels.npy           # Class label array
```

## License

MIT License — see [LICENSE](LICENSE) for details.
