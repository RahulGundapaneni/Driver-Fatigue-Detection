"""Real-time driver fatigue detection using a pretrained CNN eye classifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model


EYE_SIZE = (24, 24)
DEFAULT_THRESHOLD = 5
CASCADE_DIR = Path("haar cascade files")


def load_cascade(path: Path) -> cv2.CascadeClassifier:
    """Load a Haar cascade and fail fast with a helpful message."""
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise FileNotFoundError(
            f"Failed to load cascade from {path}. Ensure the file exists and OpenCV can access it."
        )
    return cascade


def prepare_eye_patch(eye_img: np.ndarray) -> np.ndarray | None:
    """Convert a colour eye crop into the model's grayscale input tensor."""
    if eye_img.size == 0:
        return None
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, EYE_SIZE).astype("float32") / 255.0
    return resized.reshape(1, EYE_SIZE[0], EYE_SIZE[1], 1)


def predict_eye_state(model, crop: np.ndarray | None) -> int | None:
    """Return 0 for closed, 1 for open, or None when prediction is not possible."""
    if crop is None:
        return None
    preds = model.predict(crop, verbose=0)
    return int(np.argmax(preds, axis=1)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/cnnCat2.h5", help="Path to the trained Keras model.")
    parser.add_argument("--alarm", default="alarm.wav", help="Audio file to play when drowsiness detected.")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="Frames required to trigger alarm.")
    parser.add_argument("--device", type=int, default=0, help="OpenCV video capture device index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}. Re-run training or update the path.")

    face_cascade = load_cascade(CASCADE_DIR / "haarcascade_frontalface_alt.xml")
    left_cascade = load_cascade(CASCADE_DIR / "haarcascade_lefteye_2splits.xml")
    right_cascade = load_cascade(CASCADE_DIR / "haarcascade_righteye_2splits.xml")

    model = load_model(model_path)

    mixer.init()
    sound = mixer.Sound(args.alarm)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video capture device {args.device}.")

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    thicc = 2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
            left_eyes = left_cascade.detectMultiScale(gray)
            right_eyes = right_cascade.detectMultiScale(gray)

            cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

            r_state = None
            for (x, y, w, h) in right_eyes:
                r_crop = prepare_eye_patch(frame[y : y + h, x : x + w])
                r_state = predict_eye_state(model, r_crop)
                if r_state is not None:
                    break

            l_state = None
            for (x, y, w, h) in left_eyes:
                l_crop = prepare_eye_patch(frame[y : y + h, x : x + w])
                l_state = predict_eye_state(model, l_crop)
                if l_state is not None:
                    break

            if r_state == 0 and l_state == 0:
                score += 1
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                score = max(score - 1, 0)
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f"Score: {score}", (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if score > args.threshold:
                cv2.imwrite(os.path.join(os.getcwd(), "image.jpg"), frame)
                try:
                    sound.play()
                except Exception:
                    pass

                thicc = thicc + 2 if thicc < 16 else max(thicc - 2, 2)
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

            cv2.imshow("Driver Fatigue Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
