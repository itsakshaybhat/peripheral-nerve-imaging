"""Peripheral nerve imaging pipeline.

Expected local layout:

- `data/ultrasound-nerve-segmentation/train/` for images and masks
- `models/1_unet_ner_seg.hdf5` for the trained checkpoint
- `artifacts/` for generated outputs
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv2D,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
    multiply,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RANDOM_SEED = 43
IMAGE_SIZE = (128, 128)
EPOCHS = 30
BATCH_SIZE = 32

DATASET_DIR = Path("data/ultrasound-nerve-segmentation/train")
MODEL_PATH = Path("models/1_unet_ner_seg.hdf5")
HISTORY_DIR = Path("artifacts/history")

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def dice_coef(y_true, y_pred, smooth: float = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou(y_true, y_pred, smooth: float = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f) + smooth
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f) + smooth
    return intersection / union


def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bbox_loss(y_true, y_pred):
    y_cls = K.sum(y_true, axis=-1)
    loss = K.mean(K.square(y_true - y_pred), axis=-1)
    return loss * y_cls


def cgm_loss(y_true, y_pred):
    y_tr_cls = y_true[:, 0]
    y_tr_bb = y_true[:, 1:]
    y_pr_cls = y_pred[:, 0]
    y_pr_bb = y_pred[:, 1:]

    mse = K.mean(K.square(y_tr_bb - y_pr_bb), axis=-1)
    return K.binary_crossentropy(y_tr_cls, y_pr_cls) + mse * y_tr_cls


def box_diou(b_true, b_pred):
    """Calculate DIoU for bounding boxes in xywh format."""

    b_true_xy = b_true[:, :2]
    b_true_wh = b_true[:, 2:4]
    b_true_wh_half = b_true_wh / 2.0
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[:, :2]
    b_pred_wh = b_pred[:, 2:4]
    b_pred_wh_half = b_pred_wh / 2.0
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]

    b_true_area = b_true_wh[:, 0] * b_true_wh[:, 1]
    b_pred_area = b_pred_wh[:, 0] * b_pred_wh[:, 1]
    union_area = b_true_area + b_pred_area - intersect_area

    iou_score = intersect_area / (union_area + K.epsilon())
    center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    return iou_score - center_distance / (enclose_diagonal + K.epsilon())


def load_dataset(train_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Build a dataframe with paired image and mask file paths."""

    train_dir = Path(train_dir)
    mask_paths = sorted(str(path) for path in train_dir.glob("*_mask.*"))
    image_paths = [path.replace("_mask", "") for path in mask_paths]
    return pd.DataFrame({"filename": image_paths, "mask": mask_paths})


def describe_sample(train_dir: Path = DATASET_DIR) -> None:
    """Print a quick visual sanity-check summary for the first pair."""

    image_paths = sorted(train_dir.glob("*.tif"))
    if not image_paths:
        print(f"No .tif files found in {train_dir}")
        return

    first_image = np.array(Image.open(image_paths[0]))
    mask_path = train_dir / f"{image_paths[0].stem}_mask.tif"
    if mask_path.exists():
        first_mask = np.array(Image.open(mask_path))
        first_mask = np.ma.masked_where(first_mask == 0, first_mask)
    else:
        first_mask = None

    figure, axes = plt.subplots(1, 3, figsize=(16, 12))
    axes[0].imshow(first_image, cmap="gray")
    axes[0].set_title("Image")
    if first_mask is not None:
        axes[1].imshow(first_mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[2].imshow(first_image, cmap="gray", interpolation="none")
        axes[2].imshow(first_mask, cmap="jet", interpolation="none", alpha=0.7)
        axes[2].set_title("Overlay")
    figure.tight_layout()
    plt.show()


def res_block(inputs, filter_size):
    cb1 = Conv2D(filter_size, (3, 3), padding="same", activation="relu")(inputs)
    cb2 = Conv2D(filter_size, (1, 1), padding="same", activation="relu")(inputs)
    return Add()([cb1, cb2])


def decoder_block(inputs, out_channels, depth):
    conv_kwargs = dict(
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        data_format="channels_last",
    )

    block = UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    block = Conv2D(out_channels, 3, **conv_kwargs)(block)
    block = Conv2D(out_channels, 3, **conv_kwargs)(block)
    if depth > 2:
        block = Conv2D(out_channels, 3, **conv_kwargs)(block)
    return block


def TransCGUNet(input_size=(512, 512, 1), pruned: bool = False):
    """Build the segmentation, classification, and bounding-box network."""

    inputs = Input(input_size)
    encoder = VGG16(include_top=False, weights="imagenet", input_shape=input_size)

    enc1 = encoder.get_layer(name="block1_conv1")(inputs)
    enc1 = encoder.get_layer(name="block1_conv2")(enc1)
    enc2 = MaxPooling2D(pool_size=(2, 2))(enc1)

    enc2 = encoder.get_layer(name="block2_conv1")(enc2)
    enc2 = encoder.get_layer(name="block2_conv2")(enc2)
    enc3 = MaxPooling2D(pool_size=(2, 2))(enc2)

    enc3 = encoder.get_layer(name="block3_conv1")(enc3)
    enc3 = encoder.get_layer(name="block3_conv2")(enc3)
    enc3 = encoder.get_layer(name="block3_conv3")(enc3)
    center = MaxPooling2D(pool_size=(2, 2))(enc3)

    center = Conv2D(512, (3, 3), activation="relu", padding="same", name="center1")(center)
    center = Conv2D(512, (3, 3), activation="relu", padding="same", name="center2")(center)

    cls = Conv2D(32, (3, 3), activation="relu", padding="same")(center)
    cls = Conv2D(1, (1, 1))(cls)
    cls = GlobalAveragePooling2D()(cls)
    cls = Activation("sigmoid", name="class")(cls)
    cls_reshaped = Reshape((1, 1, 1), name="reshape")(cls)

    dec3 = decoder_block(center, 256, 3)
    dec2 = decoder_block(dec3, 128, 2)
    dec1 = decoder_block(dec2, 64, 1)

    out = Conv2D(1, 1)(dec1)
    out = Activation("sigmoid", name="pre")(out)
    seg = multiply([out, cls_reshaped], name="seg")

    if pruned:
        return Model(inputs=[inputs], outputs=[out])

    model = Model(inputs=[inputs], outputs=[seg, cls])
    return add_bbox(model)


def add_bbox(model):
    bbox_branch = Conv2D(256, (3, 3), activation="relu", padding="same")(model.get_layer("center2").output)
    bbox_branch = Conv2D(256, (3, 3), activation="relu", padding="same")(bbox_branch)
    bbox_branch = MaxPooling2D(pool_size=(2, 2))(bbox_branch)
    bbox_branch = Conv2D(128, (3, 3), activation="relu", padding="same")(bbox_branch)
    bbox_branch = Conv2D(128, (3, 3), activation="relu", padding="same")(bbox_branch)
    bbox_branch = MaxPooling2D(pool_size=(2, 2))(bbox_branch)
    bbox_branch = Conv2D(64, (3, 3), activation="relu", padding="same")(bbox_branch)
    bbox_branch = Conv2D(64, (3, 3), activation="relu", padding="same")(bbox_branch)
    bbox_branch = MaxPooling2D(pool_size=(2, 2))(bbox_branch)
    bbox_branch = Conv2D(32, (3, 3), activation="relu", padding="same")(bbox_branch)
    bbox_branch = Conv2D(32, (3, 3), activation="relu", padding="same")(bbox_branch)
    bbox = Conv2D(4, (1, 1))(bbox_branch)
    bbox = GlobalAveragePooling2D()(bbox)
    bbox = Activation("sigmoid", name="bbox")(bbox)
    return Model(inputs=[model.input], outputs=[model.output[0], model.output[1], bbox])


def adjust_data(img, mask):
    img = preprocess_input(img)

    bbox = np.zeros((len(img), 4))
    for index, mask_image in enumerate(mask):
        mask_array = np.array(mask_image, dtype="uint8")
        _, thresh = cv2.threshold(mask_array, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 1, 2)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox[index, :] = x, y, w, h

    bbox /= IMAGE_SIZE[0]

    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    masks_sum = np.sum(mask, axis=(1, 2, 3)).reshape((-1, 1))
    class_label = (masks_sum != 0) + 0.0

    return img, mask, class_label, bbox


def train_generator(
    data_frame: pd.DataFrame,
    batch_size: int,
    train_path: str | None,
    aug_dict: dict,
    image_color_mode: str = "rgb",
    mask_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    mask_save_prefix: str = "mask",
    save_to_dir: str | None = None,
    target_size: tuple[int, int] = IMAGE_SIZE,
    seed: int = 1,
):
    """Generate synchronized image, mask, class, and bounding-box batches."""

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory=train_path,
        x_col="filename",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        directory=train_path,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )

    for img_batch, mask_batch in zip(image_generator, mask_generator):
        img_batch, mask_batch, class_label, bbox = adjust_data(img_batch, mask_batch)
        yield img_batch, [mask_batch, class_label, bbox]


def make_augmented_dataframe(train_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Create the dataframe used by the notebook, including flipped samples."""

    train_dir = Path(train_dir)
    train_image = []
    train_mask = sorted(str(path) for path in train_dir.glob("*_mask.*"))
    for mask_path in train_mask:
        train_image.append(mask_path.replace("_mask", ""))

    positive_mask = []
    positive_img = []
    negative_mask = []
    negative_img = []

    for mask_path, img_path in zip(train_mask, train_image):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if np.sum(mask) == 0:
            negative_mask.append(mask_path)
            negative_img.append(img_path)
        else:
            positive_mask.append(mask_path)
            positive_img.append(img_path)

    generated_img = []
    generated_mask = []
    generated_dir = Path("generated/img")
    generated_dir.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in zip(positive_img, positive_mask):
        image_name = Path(img_path).stem
        uf_img_path = generated_dir / f"{image_name}_uf.jpg"
        uf_mask_path = generated_dir / f"{image_name}_uf_mask.jpg"
        rf_img_path = generated_dir / f"{image_name}_rf.jpg"
        rf_mask_path = generated_dir / f"{image_name}_rf_mask.jpg"

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        uf_img = cv2.flip(img, 0)
        uf_mask = cv2.flip(mask, 0)
        cv2.imwrite(str(uf_img_path), uf_img)
        cv2.imwrite(str(uf_mask_path), uf_mask)

        rf_img = cv2.flip(img, 1)
        rf_mask = cv2.flip(mask, 1)
        cv2.imwrite(str(rf_img_path), rf_img)
        cv2.imwrite(str(rf_mask_path), rf_mask)

        generated_img.extend([str(uf_img_path), str(rf_img_path)])
        generated_mask.extend([str(uf_mask_path), str(rf_mask_path)])

    train_image_polished = train_image.copy()
    train_mask_polished = train_mask.copy()

    intruders_path = Path("archive/intruders.dat")
    if intruders_path.exists():
        with intruders_path.open("rb") as file_handle:
            intruders = pickle.load(file_handle)
        for intruder in intruders:
            if intruder in train_image_polished:
                index = train_image_polished.index(intruder)
                train_image_polished.pop(index)
                train_mask_polished.pop(index)

    augmented_img = generated_img + train_image_polished
    augmented_mask = generated_mask + train_mask_polished
    dataframe = pd.DataFrame({"filename": augmented_img, "mask": augmented_mask})
    return dataframe.sample(frac=1).reset_index(drop=True)


def evaluate_checkpoint(
    model_path: Path = MODEL_PATH,
    train_dir: Path = DATASET_DIR,
    batch_size: int = BATCH_SIZE,
):
    """Load the checkpoint and run a small evaluation pass if data exists."""

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    dataframe = load_dataset(train_dir)
    if dataframe.empty:
        raise ValueError(f"No image/mask pairs found in {train_dir}")

    model = load_model(
        model_path,
        custom_objects={
            "dice_coef_loss": dice_coef_loss,
            "iou": iou,
            "dice_coef": dice_coef,
            "bbox_loss": bbox_loss,
        },
    )

    generator = train_generator(dataframe, batch_size, None, {}, target_size=IMAGE_SIZE)
    results = model.evaluate(generator, steps=max(1, len(dataframe) // batch_size))
    metrics = dict(zip(model.metrics_names, results))
    return model, metrics


def print_split_summary(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        print("No data available.")
        return

    kfold = KFold(n_splits=5, shuffle=False)
    for fold_number, (train_index, test_index) in enumerate(kfold.split(dataframe), start=1):
        print(f"Fold {fold_number}: train={len(train_index)} test={len(test_index)}")


def main() -> None:
    print("Peripheral Nerve Imaging pipeline")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Model checkpoint:   {MODEL_PATH}")

    dataframe = load_dataset(DATASET_DIR)
    print(f"Paired samples found: {len(dataframe)}")

    if dataframe.empty:
        print("Add the dataset under data/ultrasound-nerve-segmentation/train/ to enable training and evaluation.")
        return

    print_split_summary(dataframe)

    if MODEL_PATH.exists():
        _, metrics = evaluate_checkpoint(MODEL_PATH, DATASET_DIR)
        print("Checkpoint metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")
    else:
        print("Checkpoint not found. Place 1_unet_ner_seg.hdf5 in models/ to run evaluation.")


if __name__ == "__main__":
    main()
