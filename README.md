# Injection-Molded Plastic Defect Detection

This project implements a deep-learning system for detecting visual defects in injection-molded plastic components.

The final system uses **EfficientNet-B0 with ImageNet transfer learning** for **multi-label classification** of four defect classes:

- Burn Mark
- Flash
- Short Shot
- Sink Mark

A Flask web prototype allows users to upload an image, view class probabilities and defect decisions, and inspect approximate Grad-CAM localization regions for detected defects.

---

## 1. Final System

| Item | Final Configuration |
|---|---|
| Architecture | EfficientNet-B0 |
| Learning approach | ImageNet transfer learning |
| Task | Multi-label classification |
| Number of classes | 4 |
| Input size | 224 × 224 |
| Loss function | BCEWithLogitsLoss |
| Class imbalance handling | Positive class weighting |
| Final decision threshold | 0.40 |
| Localization | Grad-CAM |
| Web interface | Flask |
| Final model file | `models/efficientnet_defect_v2.pth` |

Because this is a multi-label problem, one image may contain more than one defect at the same time.

---

## 2. Dataset

The final source dataset contains **975 plastic-part images**.

Only the following four target classes are used by the classifier:

1. Burn Mark
2. Flash
3. Short Shot
4. Sink Mark

### Target-Label Occurrences

| Class | Total Occurrences |
|---|---:|
| Burn Mark | 364 |
| Flash | 380 |
| Short Shot | 150 |
| Sink Mark | 323 |

### Final Leakage-Resistant Split

| Split | Images |
|---|---:|
| Training | 683 |
| Validation | 146 |
| Test | 146 |
| **Total** | **975** |

### Class Distribution

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Burn Mark | 257 | 49 | 58 |
| Flash | 266 | 61 | 53 |
| Short Shot | 111 | 19 | 20 |
| Sink Mark | 227 | 47 | 49 |

Images containing none of the four selected target defects were retained as negative examples.

---

## 3. Leakage-Resistant Dataset Construction

The original dataset split was not used directly.

A custom cleaning and splitting pipeline groups exact or strongly similar images before creating new train, validation, and test partitions. This reduces the risk of visually near-identical images appearing on opposite sides of the training/evaluation boundary.

The duplicate analysis uses:

- SHA-256 for exact duplicates
- perceptual hash (pHash)
- difference hash (dHash)
- cosine similarity of normalized image vectors
- horizontal-flip comparison

Strong near-duplicates are grouped using:

```text
pHash distance <= 4
dHash distance <= 6
cosine similarity >= 0.985
```

The resulting dataset contains:

```text
975 images
708 similarity groups
683 training images
146 validation images
146 test images
```

All images belonging to the same strong-similarity group are kept in the same split.

The generated dataset is stored under:

```text
data/final_clean/
├── train/
├── valid/
├── test/
├── split_manifest.csv
└── dataset_summary.txt
```

The original dataset under `data/raw/` is left unchanged.

---

## 4. Preprocessing and Augmentation

All images are resized to:

```text
224 × 224
```

ImageNet normalization is applied:

```text
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

Training augmentation includes:

- random horizontal flipping
- random rotation up to ±10°
- mild brightness adjustment
- mild contrast adjustment

Validation and test images are resized and normalized only.

---

## 5. Class Imbalance

Positive class weights are calculated automatically from the training split using:

```text
pos_weight = negative samples / positive samples
```

Final training weights are approximately:

| Class | Positive Training Images | `pos_weight` |
|---|---:|---:|
| Burn Mark | 257 | 1.6576 |
| Flash | 266 | 1.5677 |
| Short Shot | 111 | 5.1532 |
| Sink Mark | 227 | 2.0088 |

These weights are used with `BCEWithLogitsLoss`.

---

## 6. Training Procedure

Training uses two transfer-learning phases.

### Phase 1 — Classifier Head

The EfficientNet-B0 backbone is frozen and only the classifier head is trained.

```text
Epochs: 5
Optimizer: Adam
Learning rate: 1e-3
Batch size: 16
```

### Phase 2 — Full Fine-Tuning

The entire network is unfrozen and fine-tuned.

```text
Epochs: 10
Optimizer: Adam
Learning rate: 1e-4
Batch size: 16
```

The best checkpoint is selected using validation loss.

Final V2 best validation loss:

```text
0.6494
```

The best checkpoint is saved as:

```text
models/efficientnet_defect_v2.pth
```

---

## 7. Threshold Selection

The model produces four independent sigmoid probabilities.

A single global threshold was selected using the **validation set only**. Candidate thresholds from 0.10 to 0.90 were evaluated in steps of 0.05.

The selection criterion was:

```text
maximum validation Macro F1-score
```

The selected global threshold is:

```text
0.40
```

Validation performance at threshold 0.40:

| Metric | Result |
|---|---:|
| Macro Precision | 0.5538 |
| Macro Recall | 0.8340 |
| Macro F1 | 0.6459 |
| Micro Precision | 0.5598 |
| Micro Recall | 0.8239 |
| Micro F1 | 0.6667 |
| Exact Match | 29.45% |
| Hamming Accuracy | 75.17% |

The test set was not used for threshold selection.

---

## 8. Final Test Results

The final EfficientNet-B0 V2 model was evaluated on the held-out **146-image test set** using the locked threshold of 0.40.

### Overall Results

| Metric | Result |
|---|---:|
| Macro Precision | 0.5328 |
| Macro Recall | **0.8537** |
| Macro F1 | **0.6292** |
| Micro Precision | 0.5190 |
| Micro Recall | **0.8333** |
| Micro F1 | **0.6397** |
| Exact Match Accuracy | 23.97% |
| Hamming Accuracy | 71.06% |

### Per-Class Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Burn Mark | 0.6471 | 0.7586 | 0.6984 |
| Flash | 0.7500 | 0.8491 | 0.7965 |
| Short Shot | 0.2676 | 0.9500 | 0.4176 |
| Sink Mark | 0.4667 | 0.8571 | 0.6043 |

Short Shot achieves very high recall but relatively low precision, meaning false-positive Short Shot predictions remain an important limitation.

---

## 9. Effect of Increasing Dataset Size and Diversity

The original EfficientNet model was trained using the smaller 262-image dataset.

To test whether the larger dataset improved generalization, both the original and V2 models were evaluated on the same strict subset of images.

Any V2 test image that was an exact duplicate or strong perceptual near-duplicate of any image in the original 262-image dataset was excluded.

This left:

```text
91 strictly unseen images
```

### Strict Old-vs-V2 Comparison

| Metric | Original Model | V2 Model | Change |
|---|---:|---:|---:|
| Macro Precision | 0.4447 | **0.5501** | +0.1053 |
| Macro Recall | 0.3512 | **0.8833** | +0.5321 |
| Macro F1 | 0.3779 | **0.6645** | +0.2867 |
| Micro Precision | **0.5714** | 0.5693 | -0.0021 |
| Micro Recall | 0.4275 | **0.8779** | +0.4504 |
| Micro F1 | 0.4891 | **0.6907** | +0.2016 |
| Exact Match | 15.38% | **27.47%** | +12.09 percentage points |
| Hamming Accuracy | 67.86% | **71.70%** | +3.85 percentage points |

This comparison shows that the larger and more diverse training dataset substantially improved recall, Macro F1, Micro F1, and exact-match performance on previously unseen plastic-part images.

---

## 10. Grad-CAM Localization

The Flask prototype uses **Grad-CAM** to visualize approximate image regions that influenced each detected class prediction.

For every class whose predicted probability is greater than or equal to 0.40, the system:

1. computes a class-specific Grad-CAM heatmap;
2. identifies a strong activation region;
3. extracts an approximate contour;
4. overlays the contour and defect label on the original image.

Grad-CAM is an interpretability technique, not a true segmentation method.

The displayed region should therefore be interpreted as:

```text
an approximate region that influenced the model's prediction
```

rather than an exact physical defect boundary.

Pixel-accurate localization would require a dedicated segmentation model and pixel-level ground-truth masks.

---

## 11. Flask Prototype, Phone Camera and API

The final Flask interface supports both direct phone-camera capture and conventional file upload. On a compatible phone, the user can select **Take Photo** to open the rear camera or **Upload Existing Image** to choose a JPG/PNG from the device. Both sources use one **Analyse Image** action and the same EfficientNet-B0 V2 inference pipeline.

The interface displays probabilities for all four target defects, Detected/Clear decisions, the overall defect status, and Grad-CAM localization for detected classes. The prototype uses:

```text
Model: models/efficientnet_defect_v2.pth
Threshold: 0.40
Web server: Flask on port 5000
```

The application also exposes a prediction API:

```text
POST /api/predict
Form field: image
```

The endpoint returns JSON containing the model name, threshold, detected classes and class probabilities. API requests skip Grad-CAM to reduce inference overhead. A GET request to `/api` returns basic API information.

For phone access on the same local network, Flask binds to `0.0.0.0:5000`; the phone opens the laptop's IPv4 address, for example `http://192.168.x.x:5000`. A simple diffuse light box with a fixed camera position is recommended for consistent lighting, background, distance and viewing angle.

---

## 12. Project Structure

```text
project/
│
├── app/
│   ├── app.py
│   ├── static/
│   │   └── css/
│   │       └── style.css
│   └── templates/
│       ├── base.html
│       └── index.html
│
├── data/
│   ├── raw/
│   └── final_clean/
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── split_manifest.csv
│       └── dataset_summary.txt
│
├── experiments/
├── models/
│   ├── efficientnet_defect.pth
│   ├── efficientnet_defect_v2.pth
│   └── resnet18_baseline.pth
├── new_dataset/
├── results/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_v2.py
│   ├── optimize_threshold.py
│   ├── optimize_threshold_v2.py
│   ├── gradcam.py
│   ├── build_clean_dataset.py
│   ├── check_dataset_duplicates.py
│   ├── compare_models_fair.py
│   ├── compare_models_strict.py
│   ├── model_resnet.py
│   ├── train_resnet.py
│   ├── evaluate_resnet.py
│   └── optimize_resnet_threshold.py
├── requirements.txt
└── README.md
```

---

## 13. Environment Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Main libraries include:

- PyTorch
- torchvision
- Flask
- OpenCV
- NumPy
- pandas
- Pillow
- scikit-learn
- matplotlib

---

## 14. Running the Prototype

From the project root:

```powershell
python app\app.py
```

On the development computer, open:

```text
http://127.0.0.1:5000
```

To use the phone camera, connect the phone and laptop to the same local network, run `ipconfig` on Windows, and open the laptop IPv4 address from the phone:

```text
http://<LAPTOP-IPV4>:5000
```

The application loads:

```text
models/efficientnet_defect_v2.pth
Decision threshold: 0.40
```

The API can be inspected at:

```text
http://127.0.0.1:5000/api
```

Example PowerShell API test:

```powershell
curl.exe -X POST `
  -F "image=@C:\path\to\test_image.jpg" `
  http://127.0.0.1:5000/api/predict
```

---

## 15. Reproducing the Main Experiment

### Build the Leakage-Resistant Dataset

```powershell
python src\build_clean_dataset.py
```

### Train EfficientNet-B0 V2

```powershell
python src\train.py
```

### Optimize the Validation Threshold

```powershell
python src\optimize_threshold_v2.py
```

### Evaluate the Final Model

```powershell
python src\evaluate_v2.py
```

### Run the Strict Old-vs-V2 Comparison

```powershell
python src\compare_models_strict.py
```

---

## 16. Main Generated Outputs

```text
models/efficientnet_defect_v2.pth
results/efficientnet_v2_training_log.csv
results/efficientnet_v2_training_history.png
results/efficientnet_v2_threshold_search.csv
results/efficientnet_v2_threshold_summary.txt
results/efficientnet_v2_final_evaluation.txt
results/efficientnet_v2_confusion_matrix.png
results/strict_comparison_filter_report.csv
results/strict_old_vs_v2_comparison.txt
```

---

## 17. Limitations

Although the final dataset is substantially larger than the original dataset, it is still relatively small compared with large-scale industrial computer-vision datasets.

Short Shot has high recall but relatively low precision, so false-positive Short Shot predictions remain a weakness.

The model performs image-level classification rather than true object detection or semantic segmentation.

Grad-CAM provides approximate interpretability regions and should not be treated as pixel-accurate defect localization.

The dataset contains groups of visually similar images. A similarity-aware splitting process was therefore used to reduce leakage, but images from the same manufacturing environment may still share visual characteristics.

---

## 18. Technologies Used

- Python
- PyTorch
- torchvision
- EfficientNet-B0
- ResNet18
- Flask
- OpenCV
- Grad-CAM
- NumPy
- pandas
- Pillow
- scikit-learn
- matplotlib
- HTML
- CSS

---

## 19. Authors

Group Project — Injection-Molded Plastic Parts Defect Detection

Group Members:

1. Muzikayise Ndlovu
2. Abongile Nzama
3. Sduduzo Shandu
4. Aphiwe Sthole
