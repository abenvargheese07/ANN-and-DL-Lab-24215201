# Lab #5: CNN for Binary Image Classification (Cats vs. Dogs)

Notebook: `Lab5_CNN_Cats_vs_Dogs.ipynb`

## Overview

This notebook builds, trains, and evaluates a Convolutional Neural Network (CNN) in
TensorFlow/Keras for **binary image classification** — cats vs. dogs — using the
"Cats and Dogs Filtered" dataset (2,000 training / 1,000 validation images,
downloaded automatically from Google's hosted tutorial URL). It covers image
preprocessing, data augmentation, CNN architecture design, training, evaluation, and
interpretation of individual predictions.

## How to run

1. Open the notebook in **Google Colab** (recommended — a GPU runtime speeds up
   training significantly: `Runtime > Change runtime type > GPU`).
2. Run all cells top to bottom (`Runtime > Run all`).
3. The dataset (~68 MB) downloads and extracts automatically on first run via
   `keras.utils.get_file` — no Kaggle account or manual upload needed.
4. The notebook trains **two** CNNs (with and without data augmentation) at 20
   epochs each, so a GPU runtime is recommended; on CPU this will be noticeably
   slower but will still complete.

## Requirements

- Python 3.9+
- `tensorflow` (2.x)
- `scikit-learn`
- `numpy`, `matplotlib`, `seaborn`

All of these are pre-installed in Google Colab, so no setup is needed there. For a
local environment:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn
```

## Notebook structure

| Section | What it covers |
|---|---|
| 1. Dataset Preparation | Download/extract, explore class counts, sample images, `image_dataset_from_directory`, rescaling, augmentation pipeline |
| 2. CNN Model Development | `build_cnn()` architecture (4x Conv2D+MaxPooling → Dropout → Dense(512) → sigmoid output) |
| 3. Model Training | Trains the CNN with augmentation (primary model) and without augmentation (comparison), plots accuracy/loss curves |
| 4. Model Evaluation | Accuracy, precision, recall, F1-score, confusion matrix, with-vs-without-augmentation accuracy comparison |
| 5. Interpreting Predictions | Visualizes a grid of validation images with true label, predicted label, and confidence |
| 6. Analysis and Interpretation | Discussion of architecture choices, augmentation effect, overfitting/underfitting, failure modes, improvement ideas |
| 7. Conclusion | Summary of findings |

## Notes

- The **augmented model is the primary/final model**; the non-augmented model exists
  purely as a controlled comparison to demonstrate augmentation's regularizing
  effect, directly addressing the "apply data augmentation" and "interpret results"
  learning outcomes.
- Both CNNs share the **same architecture, optimizer (Adam, lr=1e-4), batch size
  (32), and epoch budget (20)** — only the presence/absence of augmentation differs
  between them, for a fair comparison.
- Validation data is **never augmented** — augmentation is applied only to the
  training pipeline, so validation metrics reflect performance on realistic,
  unmodified images.
- Exact accuracy/loss numbers and which failure cases appear will vary slightly by
  run (weight initialization, shuffling); the prose throughout describes the
  *typical* pattern seen on this dataset — update it to match your own printed
  output before submitting.
- To adapt this notebook to a different two-class image dataset, only Section 1
  (the download/`image_dataset_from_directory` calls) needs to change — the CNN
  architecture, training loop, and evaluation code are dataset-agnostic as long as
  the directory structure has one subfolder per class.

---

## Full write-up text (all markdown cells from the notebook)

This section reproduces every piece of explanatory/analysis text from the
notebook itself (everything except the actual code cells), in order, for quick
reference without opening the `.ipynb` file.

# Lab #5: CNN for Binary Image Classification (Cats vs. Dogs)

**Dataset:** Cats vs. Dogs (filtered subset hosted by Google, the standard
TensorFlow tutorial dataset — 2,000 training images and 1,000 validation images,
balanced between the two classes)

**Objective:** Using TensorFlow/Keras, build, train, and evaluate a Convolutional
Neural Network (CNN) for binary image classification, applying image preprocessing
and data augmentation, and interpreting the model's predictions and performance.

**Learning outcomes covered in this notebook:**
* Understand the architecture of Convolutional Neural Networks
* Implement CNNs using TensorFlow/Keras
* Apply image preprocessing and data augmentation techniques
* Train and evaluate a CNN for binary image classification
* Interpret prediction results and assess model performance

## 1. Dataset Preparation

### 1.1 Dataset Description and Source

We use the **"Cats and Dogs Filtered"** dataset, a curated subset of the original
Kaggle Cats vs. Dogs dataset, hosted by Google and widely used in official
TensorFlow/Keras tutorials. It contains:

* **2,000 training images** (1,000 cats + 1,000 dogs)
* **1,000 validation images** (500 cats + 500 dogs)

This is a **binary (two-class) image classification** problem: given a photo, predict
whether it shows a **cat (0)** or a **dog (1)**. The dataset is downloaded directly
from Google's hosted URL, so no Kaggle authentication is required in Colab.

The dataset is **perfectly balanced** (1,000 cats / 1,000 dogs in training;
500 / 500 in validation), so no class-imbalance handling (e.g. class weighting) is
required, and plain accuracy is a meaningful metric alongside precision/recall/F1.

Sample images show that photos vary in **size, lighting, pose, background
clutter, and zoom level** — this natural variability is exactly what motivates using
**data augmentation** (random flips, rotations, zooms) during training, so the model
learns features that generalize rather than memorizing specific poses/backgrounds.

**Identifying input/output:**
* **Input:** RGB images resized to a fixed **150x150** resolution (chosen to balance
  detail retention against training speed on CPU/GPU in Colab).
* **Target:** a single **binary label** — 0 for `cats`, 1 for `dogs` — inferred
  automatically from the two subfolder names, matching a **sigmoid output neuron**
  used in the CNN below.

No missing values or categorical encoding are relevant here (the "features" are raw
pixel arrays and the target is already a clean binary label), but images do need
**pixel-value rescaling**, which we apply next.

**Justification of preprocessing choices**

* **Resizing to 150x150** keeps images small enough to train quickly in Colab while
  retaining enough detail for the CNN to learn distinguishing cat/dog features.
* **Rescaling pixel values to [0, 1]** (dividing by 255) is essential for stable,
  fast gradient-based training — raw 0-255 pixel values would produce very large,
  poorly-scaled activations.
* **Data augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`,
  `RandomContrast`) is applied **only to the training set** to artificially expand
  the effective training data and reduce overfitting, since the raw dataset is
  relatively small (2,000 training images). It is **never** applied to the
  validation set, so validation performance reflects the model's behavior on
  unmodified, realistic images.
* **`.cache().prefetch()`** are standard `tf.data` performance optimizations that
  keep the GPU/CPU fed with data during training and do not change the data itself.

## 2. CNN Model Development

**Architecture:**
* Input: 150x150x3 RGB images
* **Conv block 1:** Conv2D(32 filters, 3x3, ReLU) → MaxPooling2D(2x2)
* **Conv block 2:** Conv2D(64 filters, 3x3, ReLU) → MaxPooling2D(2x2)
* **Conv block 3:** Conv2D(128 filters, 3x3, ReLU) → MaxPooling2D(2x2)
* **Conv block 4:** Conv2D(128 filters, 3x3, ReLU) → MaxPooling2D(2x2)
* Flatten
* Dropout(0.5) — regularization to further reduce overfitting
* Dense(512, ReLU)
* **Output:** Dense(1, **sigmoid**) — single neuron for binary classification

Each convolutional block **doubles (or holds) the number of filters** while pooling
**halves the spatial resolution**, a standard CNN design pattern that lets early
layers learn low-level features (edges, textures) and deeper layers learn more
abstract, higher-level features (ears, fur patterns, facial structure) with a
progressively smaller, more compact spatial representation.

**Reported architecture / training configuration**

| Item | Value |
|---|---|
| Input image size | 150 x 150 x 3 (RGB) |
| Number of classes | 2 (cat / dog) |
| Convolutional blocks | 4 (32 → 64 → 128 → 128 filters, 3x3 kernels) |
| Pooling | MaxPooling2D (2x2) after every conv block |
| Regularization | Dropout(0.5) before the dense layers |
| Dense layer | 512 neurons, ReLU |
| Output layer | 1 neuron, **sigmoid** activation |
| Loss function | Binary cross-entropy |
| Optimizer | Adam, learning rate = 1e-4 |
| Batch size | 32 |
| Epochs | 20 |

## 3. Model Training

We train the CNN **with data augmentation** as the primary model, and additionally
train a **second CNN without augmentation** (identical architecture, split, batch
size, and epoch budget) purely to *interpret the effect of augmentation* on
overfitting — directly addressing the "apply data augmentation techniques" and
"interpret results" learning outcomes.

**Interpreting the curves:** the model **without augmentation** typically shows
training accuracy climbing well above validation accuracy (and training loss
dropping well below validation loss) as epochs progress — the classic signature of
**overfitting**, since the small 2,000-image training set gets memorized. The model
**with augmentation** typically shows training and validation accuracy/loss tracking
each other more closely, because each epoch sees randomly transformed versions of
the training images, which acts as a regularizer and improves generalization to the
validation set.

## 4. Model Evaluation

We evaluate the **augmented model** (our primary/final model) on the held-out
validation set using standard binary classification metrics: **accuracy, precision,
recall, F1-score, and a confusion matrix.**

**Confusion matrix interpretation:** the diagonal cells (correct cat→cat and
dog→dog predictions) should dominate for a well-trained model. Misclassifications
typically arise from **ambiguous images** — e.g. a cat photographed at an unusual
angle, partial occlusion, or a dog breed with cat-like facial proportions — rather
than from a systematic flaw in the architecture, since both classes have equal
representation and similar visual complexity.

## 5. Interpreting Predictions on Individual Images

We visualize a batch of validation images alongside the model's predicted label,
predicted probability, and the true label, to qualitatively assess where the model
is confident/correct vs. uncertain/wrong.

**Interpretation:** correctly classified images (green titles) usually have a
prediction probability close to 0 or 1 (high confidence), reflecting clear, unambiguous
visual features. Misclassified images (red titles) tend to have probabilities closer
to 0.5, and on visual inspection are often the images with **unusual poses, poor
lighting, partial occlusion, or an animal that visually resembles the other class**
(e.g. a cat with dog-like ears due to breed, or a small dog breed with cat-like
proportions) — exactly the kind of ambiguity a human might also find difficult.

## 6. Analysis and Interpretation Summary

1. **CNN architecture rationale** — 4 convolution + pooling blocks progressively
   extract low-level (edges/textures) to high-level (shapes/facial structure)
   features while reducing spatial dimensions, which keeps the final Dense layers
   computationally manageable and focused on the most relevant learned features.
2. **Effect of data augmentation** — comparing `history_augmented` vs. `history_plain`,
   the augmented model shows a **smaller gap** between training and validation
   accuracy/loss, indicating **better generalization and reduced overfitting** on
   this relatively small (2,000-image) training set.
3. **Overfitting / underfitting** — the non-augmented model is the more likely
   candidate for overfitting (train accuracy rising well above validation accuracy);
   if both curves plateau early at a low accuracy for either model, that would signal
   underfitting, which is unlikely here given the model capacity relative to the
   dataset size and image resolution.
4. **Evaluation metrics** — accuracy alone is sufficient to gauge overall
   performance here since the classes are perfectly balanced, but precision, recall,
   and F1 (reported above) confirm there is no asymmetric bias toward over-predicting
   one class over the other.
5. **Most common failure mode** — from the confusion matrix and the individual
   prediction grid, misclassifications cluster around **visually ambiguous images**
   rather than one class being systematically harder than the other.
6. **How to improve performance further** — additional epochs with early stopping,
   a deeper/wider CNN or transfer learning from a pretrained backbone (e.g.
   MobileNetV2, VGG16), more aggressive/additional augmentation, batch
   normalization layers, or fine-tuning the learning rate schedule.

## 7. Conclusion

This lab implemented a Convolutional Neural Network for binary image classification
(cats vs. dogs) using TensorFlow/Keras. Images were resized, rescaled, and augmented
(random flips, rotations, zooms, and contrast changes) before being fed into a
4-block Conv2D/MaxPooling architecture followed by a dropout-regularized dense head
and a sigmoid output neuron. Training a second, non-augmented model under otherwise
identical conditions showed that data augmentation measurably reduces the
train/validation performance gap, demonstrating its regularizing effect on a
relatively small image dataset. The final augmented model was evaluated on the held-
out validation set using accuracy, precision, recall, F1-score, and a confusion
matrix, and individual prediction visualizations showed that most misclassifications
occur on visually ambiguous images rather than reflecting a systematic model flaw.
Overall, the lab demonstrates the full CNN workflow — architecture design,
preprocessing, augmentation, training, and interpretation — for binary image
classification.
