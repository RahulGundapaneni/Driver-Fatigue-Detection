"""Train a CNN to classify eye states (open vs. closed) for drowsiness detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


def build_model(input_shape=(24, 24, 1), num_classes: int = 2) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(1, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(1, 1)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(1, 1)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_generators(train_dir: Path, valid_dir: Path, batch_size: int, target_size: tuple[int, int]):
    train_gen = ImageDataGenerator(rescale=1.0 / 255)
    valid_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        shuffle=True,
        color_mode="grayscale",
        class_mode="categorical",
        target_size=target_size,
    )

    valid_data = valid_gen.flow_from_directory(
        valid_dir,
        batch_size=batch_size,
        shuffle=True,
        color_mode="grayscale",
        class_mode="categorical",
        target_size=target_size,
    )

    return train_data, valid_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", default="data/train", help="Directory with training images.")
    parser.add_argument("--valid_dir", default="data/valid", help="Directory with validation images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--target_size", type=int, nargs=2, default=(24, 24), help="Image height and width.")
    parser.add_argument("--output", default="models/cnnCat2.h5", help="Path to save trained model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_dir)
    valid_dir = Path(args.valid_dir)
    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError("Training/validation directories not found. Populate them before running training.")

    batch_size = args.batch_size
    target_size = tuple(args.target_size)

    train_data, valid_data = create_generators(train_dir, valid_dir, batch_size, target_size)

    steps_per_epoch = max(train_data.samples // batch_size, 1)
    validation_steps = max(valid_data.samples // batch_size, 1)

    model = build_model(input_shape=(*target_size, 1), num_classes=train_data.num_classes)

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path, overwrite=True)
    print(f"Model saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
