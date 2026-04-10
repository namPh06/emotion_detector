"""
Training pipeline to build a higher-quality classroom emotion model.

Recommended workflow:
1. Optionally pretrain on FER2013 to learn generic facial features.
2. Fine-tune on prepared DAiSEE 4-class image folders.
3. Export the best DAiSEE checkpoint plus evaluation charts.

Expected DAiSEE directory layout:
    data/daisee_4class/
        train/0 ... train/3
        val/0   ... val/3
        test/0  ... test/3

Optional FER2013 source layout:
    fer2013/
        train/angry ... train/surprise
        test/angry  ... test/surprise
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from utils.label_mapping import FER2013_TO_EMOTION, get_ordered_labels


AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the best available classroom emotion model."
    )
    parser.add_argument(
        "--daisee-dir",
        type=Path,
        required=True,
        help="Prepared DAiSEE directory with train/val/test and class folders 0-3.",
    )
    parser.add_argument(
        "--fer-dir",
        type=Path,
        default=None,
        help="Optional FER2013 root directory with train/test emotion folders.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("training_runs") / "best_model",
        help="Directory for checkpoints, reports, and exported model.",
    )
    parser.add_argument("--img-size", type=int, default=260, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--fer-epochs-head", type=int, default=6)
    parser.add_argument("--fer-epochs-finetune", type=int, default=6)
    parser.add_argument("--daisee-epochs-head", type=int, default=10)
    parser.add_argument("--daisee-epochs-finetune", type=int, default=14)
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_best_runtime_defaults() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled for GPU training.")
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not enable mixed precision: {exc}")


def ensure_split_dirs(root: Path, split_names: Iterable[str]) -> None:
    for split in split_names:
        split_path = root / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split directory: {split_path}")


def count_images_per_class(split_dir: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for class_idx in range(4):
        class_dir = split_dir / str(class_idx)
        counts[class_idx] = 0
        if class_dir.exists():
            counts[class_idx] = sum(
                1
                for path in class_dir.rglob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
    return counts


def compute_manual_class_weights(split_dir: Path, floor: float = 0.35) -> Dict[int, float]:
    counts = count_images_per_class(split_dir)
    class_ids = np.array([idx for idx, count in counts.items() if count > 0], dtype=np.int32)
    labels = []
    for class_idx, count in counts.items():
        labels.extend([class_idx] * count)

    if not labels:
        raise ValueError(f"No training images found in {split_dir}")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=np.array(labels, dtype=np.int32),
    )
    result = {int(class_id): float(weight) for class_id, weight in zip(class_ids, weights)}

    if 0 in result:
        result[0] *= 1.4
    if 2 in result:
        result[2] *= 1.15
    if 3 in result:
        result[3] *= 1.4

    for class_idx in range(4):
        result[class_idx] = max(result.get(class_idx, 1.0), floor)
    return result


def make_datasets(
    root_dir: Path, img_size: int, batch_size: int, require_test: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset | None]:
    common_args = {
        "label_mode": "categorical",
        "image_size": (img_size, img_size),
        "batch_size": batch_size,
        "seed": SEED,
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        root_dir / "train",
        shuffle=True,
        **common_args,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root_dir / "val",
        shuffle=False,
        **common_args,
    )
    test_ds = None
    test_dir = root_dir / "test"
    if require_test:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            shuffle=False,
            **common_args,
        )

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE) if test_ds is not None else None,
    )


def build_model(img_size: int, num_classes: int = 4) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.12)(x)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08)(x)
    x = tf.keras.layers.RandomContrast(0.15)(x)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    backbone.trainable = False

    x = tf.keras.layers.BatchNormalization()(backbone.output)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="emotion",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_efficientnetv2b0")
    model.backbone = backbone
    return model


def build_loss() -> tf.keras.losses.Loss:
    focal_loss = getattr(tf.keras.losses, "CategoricalFocalCrossentropy", None)
    if focal_loss is not None:
        return focal_loss(alpha=0.3, gamma=2.0)
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_loss(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def set_backbone_trainable(model: tf.keras.Model, trainable_fraction: float) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        backbone = next(
            (
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name
            ),
            None,
        )
    if backbone is None:
        raise ValueError("Could not locate EfficientNet backbone on the model.")
    backbone.trainable = True
    total_layers = len(backbone.layers)
    freeze_until = int(total_layers * (1.0 - trainable_fraction))
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    for layer in backbone.layers[freeze_until:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def build_callbacks(run_dir: Path, stage_name: str) -> list[tf.keras.callbacks.Callback]:
    checkpoint_path = run_dir / "checkpoints" / f"{stage_name}.h5"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / f"{stage_name}.csv")),
    ]


def prepare_fer2013(source_dir: Path, prepared_dir: Path) -> Path:
    if prepared_dir.exists():
        return prepared_dir

    split_aliases = {
        "train": {"train", "training"},
        "val": {"test", "validation", "val"},
    }
    print(f"[INFO] Preparing FER2013 from {source_dir}")
    for output_split in split_aliases:
        for class_idx in range(4):
            (prepared_dir / output_split / str(class_idx)).mkdir(parents=True, exist_ok=True)

    for split_name, aliases in split_aliases.items():
        split_source = None
        for child in source_dir.iterdir():
            if child.is_dir() and child.name.lower() in aliases:
                split_source = child
                break
        if split_source is None:
            raise FileNotFoundError(f"Could not find FER2013 split for {split_name} in {source_dir}")

        for class_dir in split_source.iterdir():
            if not class_dir.is_dir():
                continue
            mapped_idx = FER2013_TO_EMOTION.get(class_dir.name.lower())
            if mapped_idx is None:
                continue
            target_dir = prepared_dir / split_name / str(mapped_idx)
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                destination = target_dir / image_path.name
                if destination.exists():
                    destination = target_dir / f"{image_path.stem}_{abs(hash(str(image_path))) % 100000}{image_path.suffix.lower()}"
                shutil.copy2(image_path, destination)

    return prepared_dir


def history_to_dict(history: tf.keras.callbacks.History) -> Dict[str, list[float]]:
    return {key: [float(x) for x in values] for key, values in history.history.items()}


def plot_training_curves(histories: Dict[str, Dict[str, list[float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for stage_name, history in histories.items():
        if "accuracy" in history:
            axes[0].plot(history["accuracy"], label=f"{stage_name} train")
        if "val_accuracy" in history:
            axes[0].plot(history["val_accuracy"], linestyle="--", label=f"{stage_name} val")
        if "loss" in history:
            axes[1].plot(history["loss"], label=f"{stage_name} train")
        if "val_loss" in history:
            axes[1].plot(history["val_loss"], linestyle="--", label=f"{stage_name} val")

    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() / 2 if cm.size else 0
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                str(cm[row_idx, col_idx]),
                ha="center",
                va="center",
                color="white" if cm[row_idx, col_idx] > threshold else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_model(
    model: tf.keras.Model, test_ds: tf.data.Dataset, output_dir: Path
) -> Dict[str, object]:
    labels_vi = get_ordered_labels("vi")
    y_true_batches = []
    y_pred_batches = []

    for images, targets in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_batches.append(np.argmax(targets.numpy(), axis=1))
        y_pred_batches.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(4)),
        target_names=labels_vi,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    plot_confusion_matrix(cm, labels_vi, output_dir / "confusion_matrix.png")

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_json(data: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def describe_split(root_dir: Path) -> Dict[str, Dict[int, int]]:
    return {
        split: count_images_per_class(root_dir / split)
        for split in ("train", "val", "test")
        if (root_dir / split).exists()
    }


def train_stage(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
    run_dir: Path,
    stage_name: str,
    epochs: int,
    learning_rate: float,
) -> tf.keras.callbacks.History:
    compile_model(model, learning_rate=learning_rate)
    print(f"[INFO] Starting {stage_name} for {epochs} epochs")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=build_callbacks(run_dir, stage_name),
        verbose=1,
    )
    return history


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    enable_best_runtime_defaults()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    ensure_split_dirs(args.daisee_dir, ("train", "val", "test"))
    dataset_summary = {"daisee": describe_split(args.daisee_dir)}

    train_ds, val_ds, test_ds = make_datasets(args.daisee_dir, args.img_size, args.batch_size)
    class_weights_daisee = compute_manual_class_weights(args.daisee_dir / "train")

    model = build_model(args.img_size)
    histories: Dict[str, Dict[str, list[float]]] = {}

    if args.fer_dir is not None:
        fer_prepared_dir = prepare_fer2013(args.fer_dir, args.work_dir / "prepared_fer2013")
        fer_train_ds, fer_val_ds, _ = make_datasets(
            fer_prepared_dir,
            args.img_size,
            args.batch_size,
            require_test=False,
        )
        class_weights_fer = compute_manual_class_weights(fer_prepared_dir / "train")
        dataset_summary["fer2013"] = describe_split(fer_prepared_dir)

        history = train_stage(
            model,
            fer_train_ds,
            fer_val_ds,
            class_weights_fer,
            args.work_dir,
            "fer_head",
            args.fer_epochs_head,
            learning_rate=8e-4,
        )
        histories["fer_head"] = history_to_dict(history)

        set_backbone_trainable(model, trainable_fraction=0.25)
        history = train_stage(
            model,
            fer_train_ds,
            fer_val_ds,
            class_weights_fer,
            args.work_dir,
            "fer_finetune",
            args.fer_epochs_finetune,
            learning_rate=1e-4,
        )
        histories["fer_finetune"] = history_to_dict(history)

    history = train_stage(
        model,
        train_ds,
        val_ds,
        class_weights_daisee,
        args.work_dir,
        "daisee_head",
        args.daisee_epochs_head,
        learning_rate=7e-4,
    )
    histories["daisee_head"] = history_to_dict(history)

    set_backbone_trainable(model, trainable_fraction=0.4)
    history = train_stage(
        model,
        train_ds,
        val_ds,
        class_weights_daisee,
        args.work_dir,
        "daisee_finetune",
        args.daisee_epochs_finetune,
        learning_rate=7e-5,
    )
    histories["daisee_finetune"] = history_to_dict(history)

    final_model_path = args.work_dir / "emotion_model_daisee_best.h5"
    model.save(final_model_path)
    shutil.copy2(final_model_path, Path("models") / "emotion_model_daisee.h5")

    metrics = evaluate_model(model, test_ds, args.work_dir)
    plot_training_curves(histories, args.work_dir / "training_curves.png")
    save_json(
        {
            "class_weights": {"daisee": class_weights_daisee},
            "datasets": dataset_summary,
            "histories": histories,
            "metrics": metrics,
        },
        args.work_dir / "training_summary.json",
    )

    print("[OK] Training complete.")
    print(f"[OK] Best model exported to: {final_model_path}")
    print(f"[OK] Inference model updated at: {Path('models') / 'emotion_model_daisee.h5'}")


if __name__ == "__main__":
    main()
