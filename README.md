# 🧠 Ultrasound Peripheral Nerve Segmentation — TransCGUNet

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📌 Project Overview

A deep learning multi-task pipeline for **automatic segmentation of the Brachial Plexus nerve** in ultrasound images. The core model architecture, **TransCGUNet**, integrates a pretrained **VGG16 ImageNet encoder** with a decoder path, featuring **classification-gated mask prediction (CGM)** and **bounding box regression**.

This project was built and validated on the [Kaggle Ultrasound Nerve Segmentation Dataset](https://www.kaggle.com/c/ultrasound-nerve-segmentation).

---

## 🏗️ Model Architecture — TransCGUNet

`TransCGUNet` is a unified multi-task architecture that simultaneously performs pixel-level segmentation, image-level presence classification, and region-level bounding box localization.

```
                  Input Image (512x512x1 / 128x128x1)
                                   │
                              ┌────▼────┐
                              │  VGG16  │  ← Pretrained ImageNet Encoder
                              │ Encoder │
                              └────┬────┘
                                   │ Encoder Feature Maps (enc1, enc2, enc3)
                              ┌────▼────┐
                              │ Bottleneck │  ← Conv2D (512 filters)
                              └────┬────┘
                                   ├─────────────────────────────────────────┐
                                   │                                         │
                      ┌────────────▼───────────┐                 ┌───────────▼────────────┐
                      │     Decoder Blocks     │                 │   Classification Head  │
                      │  dec3 → dec2 → dec1    │                 │ Conv2D + GlobalAvgPool │
                      └────────────┬───────────┘                 └───────────┬────────────┘
                                   │                                         │ Class Sigmoid
                                   │ Raw Mask                                │ Probability (1x1x1)
                                   └───────────────┬─────────────────────────┘
                                                   │
                                            ┌──────▼──────┐
                                            │ Elementwise │  ← CGM Gating Gated Mask:
                                            │ Multiply    │     seg = mask * class_prob
                                            └──────┬──────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   │                                                               │
        ┌──────────▼──────────┐                                         ┌──────────▼──────────┐
        │  Segmentation Head  │                                         │  Bounding Box Head  │
        │ Binary Mask Output  │                                         │  4D Vector [x,y,w,h]│
        └─────────────────────┘                                         └─────────────────────┘
```

### Key Architectural Components
1. **VGG16 Encoder Backbone**: Extracts hierarchical visual representations initialized with ImageNet weights.
2. **Classification-Gated Masking (CGM)**: Prevents false-positive mask predictions on negative images by multiplying the raw decoder output with the classification head probability output.
3. **Bounding Box Head**: Deep spatial feature extractor terminating in a 4-element sigmoid vector predicting normalized $(x, y, w, h)$ bounding coordinates.

---

## ✨ Key Features

- 🔬 **Nerve Segmentation**: Pixel-wise binary mask prediction with classification gating.
- 🏷️ **Classification Head**: Predicts presence/absence of nerve structures in ultrasound scans.
- 📦 **Bounding Box Head**: Predicts bounding coordinates localized around the nerve with DIoU evaluation.
- 🔄 **Hybrid Data Augmentation**: Offline vertical (`_uf`) and horizontal (`_rf`) image-mask flipping plus online ImageDataGenerator geometric transformations.
- 📊 **5-Fold Cross Validation**: Evaluates generalization capability across data splits.
- 📏 **Custom Loss Functions & Metrics**: Includes Dice Coefficient, IoU, DIoU, and joint CGM loss.
- 🧹 **Intruder Cleaning**: Automated removal of corrupted or mismatched dataset samples.

---

## 📁 Project Structure

```
peripheral-nerve-imaging/
├── nerve_segmentation_pipeline.py  # Production-ready, modular Python execution pipeline
├── full_notebook_export.py         # End-to-end notebook script export for research & experimentation
├── environment.yml                 # Conda environment specification file
├── requirements.txt                # Pip package requirements file
├── models/
│   └── README.md                   # Instructions for placing 1_unet_ner_seg.hdf5 model checkpoints
└── README.md                       # Complete project documentation
```

---

## 🧪 Custom Loss Functions & Metrics

| Component | Function | Description / Formula |
|---|---|---|
| **Segmentation Loss** | `dice_coef_loss` | $1 - \text{Dice Coefficient}$ |
| **IoU Loss** | `iou_loss` | $1 - \text{Intersection over Union}$ |
| **Bounding Box Loss** | `bbox_loss` | Mask-weighted Mean Squared Error: $\text{MSE}(\mathbf{b}_{\text{true}}, \mathbf{b}_{\text{pred}}) \times y_{\text{cls}}$ |
| **CGM Joint Loss** | `cgm_loss` | $\text{BCE}(y_{\text{cls}}, \hat{y}_{\text{cls}}) + \text{MSE}(\mathbf{b}_{\text{true}}, \mathbf{b}_{\text{pred}}) \times y_{\text{cls}}$ |
| **Bounding Box Metric**| `box_diou` | Distance-IoU metric accounting for overlap area and center distance ratio |
| **Evaluation Dice** | `dice_coef` | $\frac{2 |Y \cap \hat{Y}| + \epsilon}{|Y| + |\hat{Y}| + \epsilon}$ |

---

## ⚙️ Technologies Used

- **Python 3.10** — Primary development language
- **TensorFlow / Keras** — Deep learning model architecture, training, and custom callbacks
- **VGG16** — Pretrained convolutional backbone
- **OpenCV & PIL** — Image loading, contour processing, and bounding box extraction
- **Scikit-Learn** — K-Fold cross validation and dataset splitting
- **NumPy & Pandas** — Vectorized matrix operations and tabular dataset indexing
- **Matplotlib** — Prediction visualization and overlay rendering

---

## 🔄 Data Pipeline Workflow

```
Raw Ultrasound Scans & Binary Masks
                │
                ▼
Separate Positive (Nerve Present) & Negative Masks
                │
                ▼
Offline Data Augmentation (Up-Down & Right-Left Flips)
                │
                ▼
Intruder Data Cleaning (Filter Corrupted Masks)
                │
                ▼
5-Fold Cross-Validation Splitting
                │
                ▼
Online ImageDataGenerator (Rotation, Zoom, Shift, Flip)
                │
                ▼
TransCGUNet Multi-Task Training (30 Epochs, Batch Size 32)
                │
                ▼
Evaluation & Visualization (Dice, IoU, DIoU, Mask Overlays)
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/itsakshaybhat/peripheral-nerve-imaging.git
cd peripheral-nerve-imaging
```

### 2. Set Up Environment

#### Option A: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate peripheral-nerve-imaging
```

#### Option B: Using Pip

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset & Checkpoints

1. Download the dataset from [Kaggle Ultrasound Nerve Segmentation](https://www.kaggle.com/c/ultrasound-nerve-segmentation).
2. Extract the training images into `data/ultrasound-nerve-segmentation/train/`.
3. (Optional) Place pre-trained model checkpoints into `models/1_unet_ner_seg.hdf5`.

### 4. Execute Pipeline

Run the modular pipeline script:

```bash
python nerve_segmentation_pipeline.py
```

Or run the full research notebook export script:

```bash
python full_notebook_export.py
```

---

## 📷 Visualization & Outputs

When evaluating models, predictions generate 6 qualitative comparison panels:

| Panel | Content | Description |
|---|---|---|
| **Panel 1** | Original Image | Raw grayscale ultrasound scan input |
| **Panel 2** | Original Mask | Ground truth expert nerve annotation |
| **Panel 3** | Pre-Prediction | Raw decoder segmentation output before CGM gating |
| **Panel 4** | Prediction | Final classification-gated binary segmentation mask |
| **Panel 5** | BBox Original | Ground truth nerve region bounding box |
| **Panel 6** | BBox Predicted | Model-predicted nerve bounding box |

---

## 📚 References

- **Distance-IoU Loss**: Zheng et al., *"Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"*, AAAI 2020. [arXiv:1911.08287](https://arxiv.org/abs/1911.08287)
- **Kaggle Challenge**: [Ultrasound Nerve Segmentation Dataset](https://www.kaggle.com/c/ultrasound-nerve-segmentation)
- **VGG16 Backbone**: Simonyan & Zisserman, *"Very Deep Convolutional Networks for Large-Scale Image Recognition"*, ICLR 2015.

---

## 👨‍💻 Developed By

**Akshay Bhat**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/itsakshaybhat)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/itsakshaybhat)