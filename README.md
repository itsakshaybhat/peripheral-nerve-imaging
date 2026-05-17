# 🧠 Ultrasound Nerve Segmentation — TransCGUNet

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📌 Project Overview

A deep learning pipeline for **automatic segmentation of the Brachial Plexus nerve** in ultrasound images using a custom **TransCGUNet** architecture — a VGG16-based encoder-decoder model with integrated classification, segmentation, and bounding box prediction heads.

This project was developed on the [Kaggle Ultrasound Nerve Segmentation Dataset](https://www.kaggle.com/c/ultrasound-nerve-segmentation).

---

## 🏗️ Model Architecture — TransCGUNet

```
Input Image (512x512x1)
        │
   ┌────▼────┐
   │  VGG16  │  ← Pretrained ImageNet Encoder
   │ Encoder │
   └────┬────┘
        │ Encoder Blocks (enc1, enc2, enc3)
   ┌────▼────┐
   │  Center │  ← Bottleneck (Conv2D 512)
   └────┬────┘
        │
   ┌────▼──────────────────┐
   │     Decoder Blocks    │  ← UpSampling + Conv2D
   │  dec3 → dec2 → dec1  │
   └────┬──────────────────┘
        │
   ┌────▼──────────────────────────────────┐
   │         3 Output Heads                │
   │  1. Segmentation Mask  (sigmoid)      │
   │  2. Classification     (sigmoid)      │
   │  3. Bounding Box       (sigmoid x 4)  │
   └───────────────────────────────────────┘
```

---

## ✨ Features

- 🔬 **Nerve Segmentation** — pixel-wise binary mask prediction
- 🏷️ **Classification Head** — predicts presence/absence of nerve
- 📦 **Bounding Box Head** — localizes nerve region with DIoU loss
- 🔄 **Data Augmentation** — flipping (vertical & horizontal), rotation, zoom, shear
- 📊 **5-Fold Cross Validation** — robust model evaluation
- 📏 **Custom Metrics** — Dice Coefficient, IoU
- 🧹 **Data Cleaning** — removes intruder/corrupted masks

---

## 📁 Project Structure

```
nerve-segmentation/
  ├── nerve_image_code.ipynb     ← Main Kaggle notebook
  ├── generated/
  │   └── img/                   ← Augmented images (flip up-down, right-left)
  └── README.md
```

---

## 🧪 Custom Loss Functions

| Loss Function | Description |
|---|---|
| `dice_coef_loss` | 1 - Dice Coefficient for segmentation |
| `iou_loss` | 1 - Intersection over Union |
| `bbox_loss` | MSE loss weighted by class label |
| `cgm_loss` | Binary CrossEntropy + MSE for classification + bbox |
| `box_diou` | Distance-IoU for better bounding box regression |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| `dice_coef` | Overlap between predicted and ground truth mask |
| `iou` | Intersection over Union score |
| `binary_accuracy` | Pixel-level accuracy |
| Custom Dice | `seg_dice_coef * 0.5 + class_accuracy * 0.5 + 0.03` |

---

## ⚙️ Technologies Used

- **Python** — core language
- **TensorFlow / Keras** — model building & training
- **VGG16** — pretrained ImageNet encoder backbone
- **OpenCV** — image processing, contour detection, bounding boxes
- **Scikit-learn** — KFold cross validation, train/test split
- **PIL / skimage** — image loading and resizing
- **Matplotlib** — visualization of predictions
- **NumPy / Pandas** — data handling
- **tqdm** — progress tracking

---

## 🔄 Data Pipeline

```
Raw Ultrasound Images & Masks
        │
        ▼
Separate Positive / Negative Masks
        │
        ▼
Data Augmentation (Flip Up-Down, Flip Right-Left)
        │
        ▼
Remove Intruder/Corrupted Samples
        │
        ▼
5-Fold Cross Validation Split
        │
        ▼
ImageDataGenerator (rotation, zoom, shift, flip)
        │
        ▼
TransCGUNet Training (30 Epochs, Batch Size 32)
        │
        ▼
Evaluate: Dice, IoU, Accuracy
```

---

## 🚀 How to Run

### On Kaggle (Recommended)

1. Go to [Kaggle Ultrasound Nerve Segmentation](https://www.kaggle.com/c/ultrasound-nerve-segmentation)
2. Download the dataset
3. Upload the notebook to Kaggle
4. Enable GPU accelerator
5. Click **Run All**

### Locally

```bash
# Clone the repo
git clone https://github.com/akshay-cloud-dev/nerve-segmentation.git
cd nerve-segmentation

# Install dependencies
pip install tensorflow keras opencv-python pillow scikit-learn matplotlib tqdm pandas numpy scikit-image

# Run the notebook
jupyter notebook nerve_image_code.ipynb
```

---

## 📷 Output Visualization

Each prediction displays 6 panels:

| Panel | Description |
|---|---|
| Original Image | Raw ultrasound input |
| Original Mask | Ground truth nerve mask |
| Pre-Prediction | Segmentation before CGM gating |
| Prediction | Final segmentation output |
| BBox Original | Ground truth bounding box |
| BBox Predicted | Model predicted bounding box |

---

## 📚 References

- [Distance-IoU Loss Paper](https://arxiv.org/abs/1911.08287) — DIoU/CIoU bounding box regression
- [Kaggle Ultrasound Nerve Segmentation](https://www.kaggle.com/c/ultrasound-nerve-segmentation) — dataset
- [VGG16 — ImageNet](https://keras.io/api/applications/vgg/) — pretrained encoder backbone

---

## 👨‍💻 Developed By

**Akshay Manjunath Bhat**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/akshay-bhat-engineer)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/akshay-cloud-dev)