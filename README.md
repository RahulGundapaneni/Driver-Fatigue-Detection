# Driver Fatigue Detection

Computer-vision pipeline that tracks a driver's eye state in real time. A lightweight CNN classifies each eye as open or closed, and an alarm is triggered when both eyes stay shut for multiple frames. The repository contains training code, inference scripts, and a Colab notebook documenting the full experimentation process.

## Features
- Haar cascade–based face/eye localisation followed by CNN classification.
- Pretrained weights (`models/cnnCat2.h5`) for immediate webcam demos.
- Keras training script using directory-based generators for quick retraining.
- Google Colab notebook that covers data preprocessing, model definition, training, and an alternate inference loop.

## Repository Layout
```
.
├── detection.py                # Real-time fatigue detection script
├── model.py                    # CNN training script for 24x24 grayscale eye crops
├── models/
│   ├── cnnCat2.h5              # Saved model weights
│   └── drowsiness_detection.py # Legacy webcam demo (under review)
├── haar cascade files/         # Cascade XMLs for face and eye detection
├── Final Phase/                # Colab notebook and generated assets
├── alarm.wav                   # Alert played when drowsiness detected
├── README.md
└── requirements.txt (to create via `pip freeze > requirements.txt` when ready)
```

## Getting Started
1. **Create Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not available yet, install the essentials manually:
   ```bash
   pip install tensorflow keras opencv-python pygame numpy matplotlib scikit-learn face-recognition pillow
   ```

2. **Run the Demo**
   ```bash
   python detection.py
   ```
   Press `q` to quit. Ensure your webcam is accessible and cascades/models remain in their default locations.

3. **Retrain the Model**
   - Populate `data/train` and `data/valid` with subfolders `Open` and `Closed` (or your class names).
   - Run `python model.py` to train for 15 epochs and update `models/cnnCat2.h5`.

## Data Notes
Sample datasets are not bundled. For experimentation, reference open-eye and closed-eye image archives such as CEW or the MRL Eye Dataset. Update paths in the notebook or scripts based on your storage location.

## Roadmap
- Consolidate duplicate inference scripts into a single maintained entry point.
- Refine preprocessing utilities for automated dataset preparation.
- Package the project as a pip-installable module with CLI options.
