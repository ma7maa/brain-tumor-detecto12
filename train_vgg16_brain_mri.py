"""
Train VGG16 classifier on local brain MRI data (data/brain_mri).
Run from project root:  python train_vgg16_brain_mri.py

Requires: pip install tensorflow scikit-learn tqdm
  (On Windows Store Python, if pip fails with long path error, use a venv
   under a short path e.g. C:\\bt\\brainvenv — see README.)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
IMAGE_SIZE = (224, 224)


def project_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents[:12]]:
        if (p / "data" / "brain_mri" / "train" / "glioma").is_dir():
            return p
    raise FileNotFoundError(
        "Missing data/brain_mri/train/glioma — extract archive.zip into data/brain_mri or run from project root."
    )


def split_dataset(root: Path, seed: int = 42) -> Path:
    random.seed(seed)
    data_dir = root / "data" / "brain_mri"
    work_dir = root / "data" / "brain_mri_split"
    orig_train = data_dir / "train"
    orig_test = data_dir / "test"

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (work_dir / split / cls).mkdir(parents=True, exist_ok=True)

    all_by_class: dict[str, list[str]] = {}
    for cls in CLASSES:
        train_imgs = [str(orig_train / cls / f) for f in os.listdir(orig_train / cls)]
        test_imgs = [str(orig_test / cls / f) for f in os.listdir(orig_test / cls)]
        all_by_class[cls] = train_imgs + test_imgs

    for cls in tqdm(CLASSES, desc="Split 70/15/15"):
        imgs = all_by_class[cls]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val
        parts = (
            imgs[:n_train],
            imgs[n_train : n_train + n_val],
            imgs[n_train + n_val :],
        )
        targets = (
            work_dir / "train" / cls,
            work_dir / "val" / cls,
            work_dir / "test" / cls,
        )
        for chunk, target in zip(parts, targets):
            for fpath in chunk:
                shutil.copy(fpath, target / os.path.basename(fpath))

    return work_dir


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except ValueError:
                pass
        print("GPU:", gpus)
    else:
        print("No GPU — training on CPU (slower).")


def build_callbacks(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    best = out_dir / "vgg16_brain_best.keras"
    return [
        callbacks.ModelCheckpoint(
            str(best),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Train VGG16 on brain MRI folders")
    parser.add_argument("--skip-split", action="store_true", help="Use existing data/brain_mri_split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=15, help="Epochs with frozen VGG base")
    parser.add_argument("--epochs-finetune", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = project_root()
    os.chdir(root)
    configure_gpu()

    split_root = root / "data" / "brain_mri_split"
    if not args.skip_split:
        if split_root.exists():
            shutil.rmtree(split_root)
        split_root = split_dataset(root, seed=args.seed)
    else:
        if not (split_root / "train").is_dir():
            raise FileNotFoundError("No split found. Run without --skip-split first.")

    train_dir = str(split_root / "train")
    val_dir = str(split_root / "val")
    test_dir = str(split_root / "test")

    batch = args.batch_size
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False,
    )

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    meta = {
        "class_indices": train_gen.class_indices,
        "index_to_class": {
            str(v): k for k, v in train_gen.class_indices.items()
        },
    }
    with open(models_dir / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    base_model = VGG16(
        weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(train_gen.num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    labels = train_gen.classes
    class_labels = np.unique(labels)
    cw = compute_class_weight(
        class_weight="balanced", classes=class_labels, y=labels
    )
    class_weight_dict = {int(k): float(v) for k, v in zip(class_labels, cw)}

    cb = build_callbacks(models_dir)

    print("--- Phase 1: frozen VGG base ---")
    model.fit(
        train_gen,
        epochs=args.epochs_head,
        validation_data=val_gen,
        callbacks=cb,
        class_weight=class_weight_dict,
    )

    print("--- Phase 2: fine-tune last VGG blocks ---")
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        epochs=args.epochs_finetune,
        validation_data=val_gen,
        callbacks=cb,
    )

    final_path = models_dir / "vgg16_brain_final.keras"
    model.save(final_path)
    print("Saved:", final_path)

    print("--- Test set ---")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

    with open(models_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)


if __name__ == "__main__":
    main()
