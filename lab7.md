# Lab #7: Keras MLP for Multiclass Classification

Notebook: `Lab7_MLP_Multiclass_Classification.ipynb`

## Overview

This notebook implements a Multi-Layer Perceptron (MLP) in Keras/TensorFlow to
classify wine samples into one of **3 cultivars** based on 13 chemical features
(the scikit-learn "Wine" dataset). It compares three hidden-layer activation
functions (ReLU, Sigmoid, Tanh) and three optimizers (Adam, SGD, RMSprop) across
5 trained models, then evaluates and compares them as required by the lab spec.

## How to run

1. Open the notebook in **Google Colab** (or Jupyter with the packages below installed).
2. Run all cells top to bottom (`Runtime > Run all` in Colab).
3. Training is fast — the dataset has only 178 samples, so all 5 models train in
   well under a minute on CPU.
4. A results table is written to `model_comparison_results.csv` in the working
   directory when you run the notebook.

## Requirements

- Python 3.9+
- `tensorflow` (2.x)
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

All of these are pre-installed in Google Colab, so no setup is needed there. For a
local environment:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

## Notebook structure

| Section | What it covers |
|---|---|
| 1. Dataset Preparation | Load, explore, check missing values, class distribution, scaling, encoding, train/test split |
| 2. MLP Model Development | `build_mlp()` architecture (13 → 32 → 16 → 3, softmax output) |
| 3–5. Experiments & Training | Experiment A (activation functions), Experiment B (optimizers), training curves |
| 6. Model Evaluation | Confusion matrices for all 5 models, classification report for the best model |
| 7. Comparison Table | Auto-generated table of accuracy/precision/recall/F1 across all models |
| 8. Analysis & Interpretation | Answers to the 13 required analysis questions |
| 9. Final Model Selection | Multi-criteria justification for the chosen final model |
| 10. Conclusion | Summary of findings |

## Notes

- All 5 models use the **same train/test split, batch size (16), epoch budget
  (100), and validation split (20%)** so comparisons are fair — only the
  activation function or optimizer changes between runs.
- `comparison_df` (Section 7) and `best_name` (Section 6) are computed live from
  your run, so the exact numbers — and possibly which model "wins" — may vary
  slightly between runs due to random weight initialization. The prose in
  Section 8/9 describes the *typical* pattern; update it to match your own
  printed results before submitting.
- To adapt this notebook to a different dataset, only the data-loading cell in
  Section 1 needs to change — the rest of the pipeline is dataset-agnostic.

---

## Full write-up text (all markdown cells from the notebook)

This section reproduces every piece of explanatory/analysis text from the
notebook itself (everything except the actual code cells), in order, for quick
reference without opening the `.ipynb` file.

# Lab #7: Keras MLP for Multiclass Classification

**Dataset:** Wine Category Classification (UCI Wine dataset, via `sklearn.datasets.load_wine`)

**Objective:** Implement a Multi-Layer Perceptron (MLP) using Keras/TensorFlow for a
multiclass classification problem and analyze the effect of different activation
functions, optimizers, and model configurations on classification performance.

## 1. Dataset Preparation

### 1.1 Dataset Description

The **Wine dataset** contains the results of a chemical analysis of wines grown in the
same region in Italy, derived from **three different cultivars (classes)**. There are
**178 samples** and **13 continuous numeric features** (e.g. alcohol, malic acid, ash,
flavanoids, color intensity, proline, etc.). The classification task is to predict which
of the 3 cultivars a wine sample belongs to, based on its chemical composition.

This satisfies the lab requirement of a real-world multiclass dataset with **3 or more
classes** ("Wine-quality/category classification").

No missing values are present in this dataset, so no imputation is required.
All 13 features are numeric, so **no categorical feature encoding** is needed for the
inputs. The **target variable** is already encoded as integers 0, 1, 2 corresponding to
the three wine cultivars, which we will further one-hot encode for the Keras softmax
output layer.

The classes are **reasonably balanced** (59 / 71 / 48 samples), so no special
class-imbalance handling (e.g. class weighting, SMOTE) is required, though the model
evaluation will still use macro-averaged metrics to be robust to any mild imbalance.

The histograms show that features are on **very different numeric scales**
(e.g. `proline` ranges into the hundreds, while `hue` is close to 0-1). This strongly
motivates **feature scaling/normalization** before feeding the data into a neural
network, since MLPs trained with gradient-based optimizers converge much better when
inputs are on comparable scales.

**Justification of preprocessing choices**

* **StandardScaler** (zero mean, unit variance) was used rather than min-max scaling
  because several features (e.g. `proline`, `magnesium`) contain outliers/large
  spreads; standardization is more robust to this than min-max scaling and works
  well with the ReLU/Sigmoid/Tanh activations used later.
* The scaler is **fit only on the training set** and applied to the test set to avoid
  data leakage.
* **Stratified train/test split (80/20)** preserves the relative class proportions in
  both sets, which is important given the moderate class-size differences.
* **One-hot encoding** of the target is required because the output layer uses a
  **softmax** activation with **categorical cross-entropy** loss.
* A **validation split** will be carved out of the training data during `model.fit`
  (see Section 5) to monitor generalization during training without touching the
  held-out test set.

## 2. MLP Model Development

**Architecture:**
* Input layer: 13 features
* Hidden layer 1: 32 neurons
* Hidden layer 2: 16 neurons
* Output layer: 3 neurons (one per class), **softmax** activation

The hidden-layer activation function and the optimizer are the two variables we
sweep over in Experiments A and B below. We build a reusable function so that all
experiments share the *same* architecture, splits, batch size, and epoch budget -
only the activation function or optimizer changes at a time - to ensure a fair
comparison as required by the assignment.

**Reported architecture / training configuration**

| Item | Value |
|---|---|
| Input features | 13 |
| Number of classes | 3 |
| Hidden layer 1 neurons | 32 |
| Hidden layer 2 neurons | 16 |
| Hidden layer activation | varied (ReLU / Sigmoid / Tanh) |
| Output activation | Softmax |
| Optimizer | varied (Adam / SGD / RMSprop) |
| Learning rate | Keras default for each optimizer (not manually modified) |
| Batch size | 16 |
| Epochs | 100 |
| Validation split | 20% of training data |

## 3-5. Experiments (Activation Functions & Optimizers), Training, and Curves

We define a single training/evaluation helper so **Experiment A** (activation
functions, optimizer fixed to Adam) and **Experiment B** (optimizers, activation
fixed to ReLU) use an identical training procedure, the same train/test split, the
same batch size (16), the same number of epochs (100), and the same 20% validation
split - the only requirement for a fair comparison.

### Experiment A - Hidden-Layer Activation Functions (optimizer fixed to Adam)

### Experiment B - Optimizer Comparison (activation fixed to ReLU)

Model 4 (ReLU, SGD) and Model 5 (ReLU, RMSprop) are new; Model 1 (ReLU, Adam) from
Experiment A is reused as the third point of comparison for this experiment so that
all three optimizers are compared under an identical ReLU architecture.

### Training / Validation Curves

For each model we plot Training vs. Validation Accuracy and Training vs. Validation
Loss across epochs, to visually inspect convergence speed, overfitting, underfitting,
and stability.

**Reading the curves:**
* **ReLU** models (1, 4, 5) tend to converge fastest and reach the lowest training
  loss, since ReLU does not saturate for positive inputs and keeps gradients large.
* **Sigmoid** (Model 2) typically converges more slowly and can plateau earlier due to
  vanishing gradients in the saturated regions of the sigmoid curve, especially with a
  small, moderately-sized network like this one.
* **Tanh** (Model 3) usually sits between ReLU and Sigmoid - zero-centered outputs
  help gradient flow relative to sigmoid, but it can still saturate for large
  activations.
* Among optimizers, **SGD** (Model 4) without momentum converges the slowest and is
  the most likely to show a still-improving/noisy curve after 100 epochs (a sign of
  under-convergence rather than a modeling problem), while **Adam** and **RMSprop**
  (Models 1 and 5) adapt the learning rate per-parameter and converge noticeably
  faster and more smoothly.
* If any curve shows validation loss rising while training loss keeps falling, that
  is the signature of **overfitting**; if both curves plateau early at a high loss,
  that indicates **underfitting**. On this dataset (small, fairly separable, 13
  features), all models are expected to reach a good fit, with SGD being the most
  likely to underfit within the fixed 100-epoch budget.

## 6. Model Evaluation - Confusion Matrices

We compute the confusion matrix for every model on the **unseen test set**, using
**macro-averaged** Precision/Recall/F1 (appropriate here since the three classes are
of broadly similar size, and macro-averaging treats each class equally rather than
letting the larger class dominate).

**Confusion matrix interpretation (typical pattern on this dataset):**
* The **class_0** and **class_1** cultivars are usually predicted very accurately, as
  their chemical profiles (e.g. flavanoid and phenol content) are quite distinct.
* **class_1 vs. class_2** is the pair most frequently confused for weaker models
  (e.g. Sigmoid, or under-converged SGD), because a few samples of these cultivars
  overlap in features such as color intensity and hue.
* Misclassifications generally arise from **borderline chemical profiles** between
  cultivars and from a **model not having converged enough** (e.g. Sigmoid/SGD within
  the fixed epoch budget), rather than from label noise, since the Wine dataset is
  a clean, well-curated dataset.

## 7. Comparison Table

## 8. Analysis and Interpretation

*(Run the cells above first - the printed numbers referenced below come directly
from the `comparison_df` table and the plots produced earlier. Replace the
illustrative statements with the exact numeric values you observe in your run.)*

1. **Which activation function performed best? Why?** — Compare Models 1-3 in
   `comparison_df`. ReLU (Model 1) typically achieves the highest test accuracy and
   F1-score because it avoids vanishing gradients and trains fastest on this
   relatively small, well-scaled, non-heavily-noisy dataset.
2. **Which optimizer performed best? Why?** — Compare Models 1, 4, 5. Adam and
   RMSprop (adaptive learning rates, momentum-based) typically outperform plain SGD
   within the fixed 100-epoch budget, converging faster and reaching lower loss.
3. **Best activation + optimizer combination** — Typically **ReLU + Adam** (Model 1),
   combining fast gradient flow with adaptive step sizes.
4. **Did activation function significantly affect convergence?** — Yes: ReLU
   converges fastest, Sigmoid slowest, Tanh in between (see the overlaid validation
   curves).
5. **Did the optimizer affect convergence speed?** — Yes: Adam/RMSprop converge in
   noticeably fewer epochs than vanilla SGD.
6. **Best validation performance** — Read off the model with the highest `val_acc` /
   lowest `val_loss` in `comparison_df`.
7. **Best test performance** — Read off the model with the highest `test_accuracy` in
   `comparison_df` (`best_name` computed above).
8. **Training vs. testing performance gap** — A small gap (a few percentage points)
   indicates good generalization; a large gap would indicate overfitting - check
   `train_acc` vs. `test_accuracy` per row.
9. **Overfitting or underfitting?** — Inspect the loss curves: diverging
   train/validation loss = overfitting; both curves plateauing high = underfitting
   (most likely for the SGD model within 100 epochs).
10. **Most frequently misclassified classes** — Read off the confusion matrices;
    typically class_1 and class_2 show the most confusion.
11. **Possible reasons for misclassification** — Overlapping feature ranges between
    cultivars (borderline chemical composition) and, for weaker optimizers/activations,
    insufficient convergence within the fixed epoch budget.
12. **What could improve performance?** — More epochs or early stopping tuned per
    model, a learning-rate schedule, dropout/regularization if overfitting appears,
    hyperparameter tuning (hidden units, batch size), or k-fold cross-validation for
    a more robust estimate given the small dataset size (178 samples).
13. **Final model selection and why** — See Section 9 below.

### How dataset characteristics influenced performance
* **Number of classes (3)** — a small number of classes keeps the softmax output
  layer simple and the classification problem tractable for a small MLP.
* **Class balance** — fairly balanced classes (59/71/48) meant macro-averaged
  metrics closely track overall accuracy, and no resampling was required.
* **Feature distribution / scale differences** — very different feature scales (e.g.
  `proline` vs. `hue`) made **StandardScaler** essential; without it, gradient-based
  optimizers would converge poorly or unevenly across features.
* **Non-linearity / overlap** — mild overlap between class_1 and class_2 in a few
  chemical features is the main source of residual misclassification.
* **Small training sample size (178 total)** — makes the model relatively prone to
  variance between runs; a validation split and, ideally, cross-validation help guard
  against over-optimistic single-split estimates.

## 9. Final Model Selection

Based on `comparison_df`, the final model is selected by weighing **test accuracy,
precision/recall/F1, confusion-matrix quality, convergence stability (from the
loss/accuracy curves), and model complexity** — not test accuracy alone, as required.

In practice on this dataset, **Model 1 (ReLU hidden activation + Adam optimizer)**
is generally the recommended final model because it:
* Reaches the highest (or tied-highest) test accuracy and macro-F1 score;
* Converges quickly and smoothly, with training and validation curves tracking each
  other closely (no strong overfitting signal);
* Uses a compact architecture (32 → 16 hidden neurons) that is not needlessly complex
  for a 13-feature, 3-class problem;
* Produces a confusion matrix with the fewest off-diagonal (misclassified) samples.

*(Confirm this against your own printed `best_name` / `comparison_df` output, and
adjust the justification if a different model wins in your specific run — results can
vary slightly run to run due to weight initialization.)*

## 10. Conclusion

This lab implemented an MLP for multiclass classification of wine cultivars using
Keras/TensorFlow. After standardizing the 13 numeric features and one-hot encoding
the 3-class target, a 2-hidden-layer MLP (32 → 16 neurons) with a softmax output was
trained under 5 configurations: three hidden-layer activation functions (ReLU,
Sigmoid, Tanh) with Adam fixed, and three optimizers (Adam, SGD, RMSprop) with ReLU
fixed. Across all experiments, ReLU combined with an adaptive optimizer (Adam or
RMSprop) gave the fastest, most stable convergence and the strongest test-set
performance, while Sigmoid and plain SGD lagged in convergence speed within the
fixed epoch budget. The confusion matrices showed that most misclassifications occur
between the two more chemically similar cultivars. Overall, the experiments
confirm that both the choice of activation function and the choice of optimizer
meaningfully affect an MLP's convergence behavior and final classification
performance, even when the underlying architecture and data split are held constant.
