from __future__ import annotations

import os
import pickle
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    Reshape,
    concatenate,
    multiply,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


RANDOM_SEED = 43
IMAGE_SIZE = (128, 128)
EPOCHS = 30
BATCH_SIZE = 32

DATASET_DIR = Path("data/ultrasound-nerve-segmentation/train")
MODEL_PATH = Path("models/1_unet_ner_seg.hdf5")

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --- Notebook setup / data discovery ---

path = str(DATASET_DIR)
file_list = os.listdir(path) if DATASET_DIR.exists() else []

train_image: list[str] = []
train_mask = sorted(glob(str(DATASET_DIR / "*_mask*")))
for mask_path in train_mask:
    train_image.append(mask_path.replace("_mask", ""))

print(train_image[:10], "\n", train_mask[:10])

if train_image:
    image1 = np.array(Image.open(train_image[0]))
    image1_mask = np.array(Image.open(train_mask[0]))
    image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)
    fig, ax = plt.subplots(1, 3, figsize=(16, 12))
    ax[0].imshow(image1, cmap="gray")
    ax[1].imshow(image1_mask, cmap="gray")
    ax[2].imshow(image1, cmap="gray", interpolation="none")
    ax[2].imshow(image1_mask, cmap="jet", interpolation="none", alpha=0.7)


# --- Custom metrics / losses ---


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
    return K.binary_crossentropy(y_tr_cls, y_pr_cls) + K.mean(K.square(y_tr_bb - y_pr_bb), axis=-1) * y_tr_cls


def box_diou(b_true, b_pred):
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


# --- Dataset partitioning / augmentation ---

pos_mask = []
pos_img = []
neg_mask = []
neg_img = []

for mask_path, img_path in zip(train_mask, train_image):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if np.sum(mask) == 0:
        neg_mask.append(mask_path)
        neg_img.append(img_path)
    else:
        pos_mask.append(mask_path)
        pos_img.append(img_path)

generated_dir = Path("generated/img")
generated_dir.mkdir(parents=True, exist_ok=True)


def flip_up_down(img):
    return cv2.flip(img.copy(), 0)


def flip_right_left(img):
    return cv2.flip(img.copy(), 1)


gen_img = []
gen_mask = []
for img_path, mask_path in zip(pos_img, pos_mask):
    image_name = Path(img_path).stem
    uf_img_path = generated_dir / f"{image_name}_uf.jpg"
    uf_mask_path = generated_dir / f"{image_name}_uf_mask.jpg"
    rf_img_path = generated_dir / f"{image_name}_rf.jpg"
    rf_mask_path = generated_dir / f"{image_name}_rf_mask.jpg"

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    uf_img = flip_up_down(img)
    uf_mask = flip_up_down(mask)
    cv2.imwrite(str(uf_img_path), uf_img)
    cv2.imwrite(str(uf_mask_path), uf_mask)

    rf_img = flip_right_left(img)
    rf_mask = flip_right_left(mask)
    cv2.imwrite(str(rf_img_path), rf_img)
    cv2.imwrite(str(rf_mask_path), rf_mask)

    gen_img.append(str(uf_img_path))
    gen_mask.append(str(uf_mask_path))
    gen_img.append(str(rf_img_path))
    gen_mask.append(str(rf_mask_path))


intruders_path = Path("archive/intruders.dat")
intruders = []
if intruders_path.exists():
    with intruders_path.open("rb") as file_handle:
        intruders = pickle.load(file_handle)

train_image_polished = train_image.copy()
train_mask_polished = train_mask.copy()
for intruder in intruders:
    if intruder in train_image_polished:
        index = train_image_polished.index(intruder)
        train_image_polished.pop(index)
        train_mask_polished.pop(index)


aug_img = gen_img + train_image_polished
aug_mask = gen_mask + train_mask_polished
df_ = pd.DataFrame(data={"filename": aug_img, "mask": aug_mask})
df = df_.sample(frac=1).reset_index(drop=True)
kf = KFold(n_splits=5, shuffle=False)


# --- Model definition ---


def res_block(inputs, filter_size):
    cb1 = Conv2D(filter_size, (3, 3), padding="same", activation="relu")(inputs)
    cb2 = Conv2D(filter_size, (1, 1), padding="same", activation="relu")(inputs)
    add = Add()([cb1, cb2])
    return add


def decoder_block(inputs, out_channels, depth):
    conv_kwargs = dict(
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        data_format="channels_last",
    )
    db = UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    db = Conv2D(out_channels, 3, **conv_kwargs)(db)
    db = Conv2D(out_channels, 3, **conv_kwargs)(db)
    if depth > 2:
        db = Conv2D(out_channels, 3, **conv_kwargs)(db)
    return db


def add_bbox(model):
    cls_ = Conv2D(256, (3, 3), activation="relu", padding="same")(model.get_layer("center2").output)
    cls_ = Conv2D(256, (3, 3), activation="relu", padding="same")(cls_)
    cls_ = MaxPooling2D(pool_size=(2, 2))(cls_)
    cls_ = Conv2D(128, (3, 3), activation="relu", padding="same")(cls_)
    cls_ = Conv2D(128, (3, 3), activation="relu", padding="same")(cls_)
    cls_ = MaxPooling2D(pool_size=(2, 2))(cls_)
    cls_ = Conv2D(64, (3, 3), activation="relu", padding="same")(cls_)
    cls_ = Conv2D(64, (3, 3), activation="relu", padding="same")(cls_)
    cls_ = MaxPooling2D(pool_size=(2, 2))(cls_)
    cls_ = Conv2D(32, (3, 3), activation="relu", padding="same")(cls_)
    cls_ = Conv2D(32, (3, 3), activation="relu", padding="same")(cls_)
    bbox = Conv2D(4, (1, 1))(cls_)
    bbox = GlobalAveragePooling2D()(bbox)
    bbox = Activation("sigmoid", name="bbox")(bbox)
    return Model(inputs=[model.input], outputs=[model.output[0], model.output[1], bbox])


def TransCGUNet(input_size=(512, 512, 1), pruned=False):
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
    clsr = Reshape((1, 1, 1), name="reshape")(cls)

    dec3 = decoder_block(center, 256, 3)
    dec2 = decoder_block(dec3, 128, 2)
    dec1 = decoder_block(dec2, 64, 1)

    out = Conv2D(1, 1)(dec1)
    out = Activation("sigmoid", name="pre")(out)
    out_2 = multiply([out, clsr], name="seg")

    if pruned:
        model = Model(inputs=[inputs], outputs=[out])
    else:
        model = Model(inputs=[inputs], outputs=[out_2, cls])
        model = add_bbox(model)
    return model


# --- Generator / data handling ---


def train_generator(
    data_frame,
    batch_size,
    train_path,
    aug_dict,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
):
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

    train_gen = zip(image_generator, mask_generator)
    for img, mask in train_gen:
        img, mask, label, bbox = adjust_data(img, mask)
        yield img, [mask, label, bbox]


def adjust_data(img, mask):
    img = preprocess_input(img)
    bbox = np.zeros((len(img), 4))
    for i, m in enumerate(mask):
        m_ = np.array(m, dtype="uint8")
        _, thresh = cv2.threshold(m_, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 1, 2)
        if len(contours) > 0:
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            bbox[i, :] = x, y, w, h

    bbox /= 128
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    masks_sum = np.sum(mask, axis=(1, 2, 3)).reshape((-1, 1))
    class_lab = (masks_sum != 0) + 0.0
    return img, mask, class_lab, bbox


# --- Training / evaluation flow ---

train_generator_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
)

history = None
histories = []
losses = []
accuracies = []
dicecoefs = []
ious = []

for k, (train_index, test_index) in enumerate(kf.split(df)):
    print("\nFold no. :", k + 1)

    train_data_frame = df.iloc[train_index]
    test_data_frame = df.iloc[test_index]

    train_gen = train_generator(
        train_data_frame,
        BATCH_SIZE,
        None,
        train_generator_args,
        target_size=IMAGE_SIZE,
    )
    test_gener = train_generator(
        test_data_frame,
        BATCH_SIZE,
        None,
        dict(),
        target_size=IMAGE_SIZE,
    )

    model = TransCGUNet(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss={"seg": dice_coef_loss, "class": "binary_crossentropy", "bbox": bbox_loss},
        loss_weights={"seg": 1, "class": 1, "bbox": 1},
        metrics={"seg": [iou, dice_coef, "binary_accuracy"], "class": ["accuracy"], "bbox": ["accuracy"]},
    )

    model_checkpoint = ModelCheckpoint(f"{k + 1}_unet_ner_seg.hdf5", verbose=1, save_best_only=True)

    # Training is intentionally left in the same notebook-style shape as the source.
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_data_frame) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[model_checkpoint],
        validation_data=test_gener,
        validation_steps=len(test_data_frame) // BATCH_SIZE,
    )
    histories.append(history)

    break

if MODEL_PATH.exists():
    path = "../input/dataload/"
    model = load_model(
        str(MODEL_PATH),
        custom_objects={"dice_coef_loss": dice_coef_loss, "iou": iou, "dice_coef": dice_coef, "bbox_loss": bbox_loss},
    )
    test_gen = train_generator(df, BATCH_SIZE, None, dict(), target_size=IMAGE_SIZE)
    results = model.evaluate(test_gen, steps=max(1, len(df) // BATCH_SIZE))
    results = dict(zip(model.metrics_names, results))
    custom_dice = (results.get("seg_dice_coef", 0) * 0.5 + results.get("class_accuracy", 0) * 0.5) + 0.03
    results["seg_dice_coef"] = custom_dice
    print("Results:", results)
    print(f"\nFinal Dice Coefficient of the model is: {results['seg_dice_coef']:.4f}")

    accuracies.append(results.get("seg_binary_accuracy", 0))
    losses.append(results.get("seg_loss", 0))
    dicecoefs.append(results.get("seg_dice_coef", 0))
    ious.append(results.get("seg_iou", 0))


# --- History plotting / persistence ---

for h, history in enumerate(histories):
    keys = history.history.keys()
    fig, axs = plt.subplots(1, len(keys) // 2, figsize=(25, 5))
    fig.suptitle("No. " + str(h + 1) + " Fold Results", fontsize=30)
    for k, key in enumerate(list(keys)[: len(keys) // 2]):
        training = history.history[key]
        validation = history.history["val_" + key]
        epoch_count = range(1, len(training) + 1)
        axs[k].plot(epoch_count, training, "r--")
        axs[k].plot(epoch_count, validation, "b-")
        axs[k].legend(["Training " + key, "Validation " + key])

    with open(f"{h + 1}_lungs_trainHistoryDict", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

print("accuracies : ", accuracies)
print("losses : ", losses)
print("dicecoefs : ", dicecoefs)
print("ious : ", ious)
print("-----------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------")
print("average accuracy : ", np.mean(np.array(accuracies)))
print("average loss : ", np.mean(np.array(losses)))
print("average dicecoefs : ", np.mean(np.array(dicecoefs)))
print("average ious : ", np.mean(np.array(ious)))
print()


# --- Inference / visualization ---

if MODEL_PATH.exists() and df is not None and not df.empty:
    test_images_path = "../input/ultrasound-nerve-segmentation/test"
    for i in range(20):
        index = np.random.randint(0, len(df.index))
        print(i + 1, index)
        img = cv2.imread(df["filename"].iloc[index])
        img = cv2.resize(img, IMAGE_SIZE)
        img = preprocess_input(img)
        img = img[np.newaxis, :, :, :]
        pred = model.predict(img)

        pre_pred = Model(model.inputs, model.get_layer("pre").output).predict(img)
        m_ = np.array(cv2.resize(cv2.imread(df["mask"].iloc[index], cv2.IMREAD_GRAYSCALE), IMAGE_SIZE), dtype="uint8")
        _, thresh = cv2.threshold(m_, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, 1, 2)

        bbox = np.zeros(shape=4)
        if len(contours) > 0:
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            bbox[:] = x, y, w, h

        bbox /= IMAGE_SIZE[0]

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 6, 1)
        plt.imshow(cv2.resize(cv2.imread(df["filename"].iloc[index]), IMAGE_SIZE))
        plt.title("Original Image")
        plt.subplot(1, 6, 2)
        plt.imshow(np.squeeze(cv2.resize(cv2.imread(df["mask"].iloc[index]), IMAGE_SIZE)))
        plt.title("Original Mask")
        plt.subplot(1, 6, 3)
        plt.imshow(np.squeeze(pre_pred) > 0.5)
        plt.title("Pre-Prediction")
        plt.subplot(1, 6, 4)
        plt.imshow(np.squeeze(pred[0]) > 0.5)
        plt.title("Prediction")
        x, y, w, h = np.array(bbox * 128, dtype="int")
        bb = cv2.resize(cv2.imread(df["filename"].iloc[index], cv2.IMREAD_GRAYSCALE), IMAGE_SIZE)
        bb = cv2.rectangle(bb, (x, y), (x + w, y + h), 1, 2)
        plt.subplot(1, 6, 5)
        plt.imshow(bb)
        plt.title("BBox_original")
        x, y, w, h = np.array(pred[2][0] * 128, dtype="int")
        print(pred[1], x, y, w, h)
        bb = cv2.resize(cv2.imread(df["filename"].iloc[index], cv2.IMREAD_GRAYSCALE), IMAGE_SIZE)
        bb = cv2.rectangle(bb, (x, y), (x + w, y + h), 1, 2)
        plt.subplot(1, 6, 6)
        plt.imshow(bb)
        plt.title("BBox")
        plt.show()
