# Lab #6: Keras MLP for Regression

Notebook: `Lab6_MLP_Regression.ipynb`

## Overview

This notebook implements a Multi-Layer Perceptron (MLP) in Keras/TensorFlow to
predict **median house value** for California districts using the scikit-learn
**California Housing** dataset (8 numeric features, ~20,600 samples). It compares
three hidden-layer activation functions (ReLU, Sigmoid, Tanh) and three regression
loss functions (MSE, MAE, Huber) across 5 trained models, then evaluates and
compares them as required by the lab spec.

## How to run

1. Open the notebook in **Google Colab** (or Jupyter with the packages below installed).
2. Run all cells top to bottom (`Runtime > Run all` in Colab).
3. The dataset downloads automatically via scikit-learn on first run (~20,640 rows);
   all 5 models train in under a couple of minutes on CPU.
4. A results table is written to `regression_model_comparison_results.csv` in the
   working directory when you run the notebook.

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
| 1. Dataset Preparation | Load, explore, check missing values, target distribution, scaling, train/test split |
| 2. MLP Model Development | `build_mlp()` architecture (8 → 64 → 32 → 1, linear output) |
| 3–4. Activation & Loss Experiments | Experiment A (activation functions), Experiment B (loss functions) |
| 5. Model Training / Loss Curves | Training vs. validation loss plots for all 5 models |
| 6. Model Evaluation | Predicted-vs-actual scatter plots for all 5 models |
| 7. Comparison Table | Auto-generated table of MAE/RMSE/R²/MSE across all models |
| 8. Analysis & Interpretation | Answers to all required analysis questions |
| 9. Final Model Selection | Multi-criteria justification for the chosen final model |
| 10. Conclusion | Summary of findings |

## Notes

- All 5 models use the **same train/test split, optimizer (Adam), batch size (32),
  epoch budget (100), and validation split (20%)** so comparisons are fair — only
  the activation function or loss function changes between runs.
- `comparison_df` (Section 7) is computed live from your run, so the exact numbers —
  and possibly which model "wins" — may vary slightly between runs due to random
  weight initialization. The prose in Section 8/9 describes the *typical* pattern;
  update it to match your own printed results before submitting.
- The target variable (`MedHouseVal`) is capped at 5.0 ($500,000) in the source
  data — a known artifact of this dataset that behaves like a mild outlier cluster
  and is used throughout the notebook to justify comparing MSE against the more
  outlier-robust MAE/Huber losses.
- To adapt this notebook to a different dataset, only the data-loading cell in
  Section 1 needs to change — the rest of the pipeline is dataset-agnostic.

---

## Full write-up text (all markdown cells from the notebook)

This section reproduces every piece of explanatory/analysis text from the
notebook itself (everything except the actual code cells), in order, for quick
reference without opening the `.ipynb` file.

# Lab #6: Keras MLP for Regression

**Dataset:** California Housing Price Prediction (`sklearn.datasets.fetch_california_housing`)

**Objective:** Implement a Multi-Layer Perceptron (MLP) using Keras/TensorFlow for a
regression problem and analyze the effect of different activation functions and loss
functions on model performance.

## 1. Dataset Preparation

### 1.1 Dataset Description and Source

The **California Housing dataset** (derived from the 1990 U.S. Census, distributed via
scikit-learn) contains **20,640 samples**, each describing a California district by
**8 numeric input features** — median income, house age, average rooms, average
bedrooms, population, average occupancy, latitude, and longitude. The **continuous
target variable** is the **median house value** for the district (in units of
$100,000). This is a classic **house-price prediction** regression problem, matching
the lab's suggested problem types.

* **Input features (8):** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`,
  `AveOccup`, `Latitude`, `Longitude`
* **Target variable:** `MedHouseVal` (median house value, continuous, in $100,000s)

No missing values are present in this version of the dataset, so no imputation
is required. All 8 features are numeric (no categorical columns), so **no categorical
encoding** is needed for the inputs.

The target is **right-skewed** and is **capped at 5.0** ($500,000) — a known
artifact of this dataset where the census bureau top-coded high values. This means a
small subset of samples sit at exactly the cap, which behaves like a mild outlier
cluster the model must handle. This observation directly motivates comparing **MSE**
(sensitive to large errors) against **MAE** and **Huber loss** (more robust to
outliers) later in the lab.

Feature scales differ enormously (e.g. `Population` is in the thousands while
`AveOccup`/`AveRooms` are small decimals, and a handful of extreme outliers exist in
`AveRooms`, `AveBedrms`, `AveOccup`, and `Population`). This motivates **feature
scaling** before training the MLP, since gradient-based optimizers converge poorly
when input features are on very different scales.

**Justification of preprocessing choices**

* **StandardScaler** (zero mean, unit variance) was applied because the 8 features
  span very different numeric ranges (e.g. `Population` in the thousands vs. `AveOccup`
  as a small decimal); without scaling, features with larger raw magnitudes would
  dominate the gradient updates.
* The scaler is **fit only on the training set** and applied to the test set to avoid
  data leakage.
* The **target variable is left unscaled** (in its native $100,000 units) so that the
  reported MAE/RMSE are directly interpretable in real house-value units; this is a
  valid and common choice for regression with Keras as long as the loss/metrics are
  interpreted in the same units.
* An **80/20 train/test split** is used, and a **validation split** is additionally
  carved out of the training data during `model.fit` (see Section "Model Training")
  to monitor generalization during training without touching the held-out test set.

## 2. MLP Model Development

**Architecture:**
* Input layer: 8 features
* Hidden layer 1: 64 neurons
* Hidden layer 2: 32 neurons
* Output layer: 1 neuron, **linear** activation (standard for regression, so
  predictions are not artificially bounded)

The hidden-layer activation function and the loss function are the two variables we
sweep over in the experiments below. A reusable builder function ensures every
experiment shares the *same* architecture, data split, batch size, and epoch budget —
only the activation function or loss function changes at a time — for a fair
comparison as required by the assignment.

**Reported architecture / training configuration**

| Item | Value |
|---|---|
| Input features | 8 |
| Output neurons | 1 (linear activation) |
| Hidden layer 1 neurons | 64 |
| Hidden layer 2 neurons | 32 |
| Hidden layer activation | varied (ReLU / Sigmoid / Tanh) |
| Loss function | varied (MSE / MAE / Huber) |
| Optimizer | Adam (kept constant across all experiments) |
| Learning rate | Keras default for Adam (not manually modified) |
| Batch size | 32 |
| Epochs | 100 |
| Validation split | 20% of training data |

## 3. Experiment with Activation Functions

We compare **ReLU**, **Sigmoid**, and **Tanh** as the hidden-layer activation
function, with the **loss function fixed to MSE** and every other setting
(architecture, split, batch size, epochs, optimizer) held constant, so only the
activation function varies.

## 4. Experiment with Loss Functions

We compare **MSE**, **MAE**, and **Huber loss** as the training loss, with the
**hidden-layer activation fixed to ReLU** and every other setting held constant.

**Why these three losses are appropriate for this dataset:**
* **MSE (Mean Squared Error)** — squares the error, so it penalizes large errors
  heavily. It is the standard default for regression and works well when errors are
  roughly Gaussian, but it can be overly sensitive to the top-coded/outlier house
  values noted above.
* **MAE (Mean Absolute Error)** — penalizes all errors linearly, making it more
  **robust to outliers** than MSE. Given the target's right-skew and the capped
  values at 5.0, MAE is expected to be less distorted by those extreme points.
* **Huber loss** — behaves like MSE for small errors and like MAE for large errors
  (controlled by a delta threshold), combining smooth gradients near zero error with
  robustness to outliers — a natural middle ground for this dataset's mix of
  well-behaved and outlier-prone samples.

### Running Experiment A - Activation Functions (loss fixed to MSE)

### Running Experiment B - Loss Functions (activation fixed to ReLU)

Model 1 (ReLU, MSE) from Experiment A is reused as one of the three loss-function
comparison points, alongside two new models trained with MAE and Huber loss, so all
three losses are compared under an identical ReLU architecture.

## 5. Model Training - Loss Curves

For each model we plot Training vs. Validation loss across epochs to inspect
convergence speed, overfitting, underfitting, and stability. Note that because
different models are trained with different loss functions (MSE vs. MAE vs. Huber),
the **raw loss values are not directly comparable across loss functions** — each
model's curve should be read for its own convergence *shape*, while the
**test-set MAE/RMSE/R² in the comparison table** (common metrics computed the same
way for every model) are what should be used for cross-model comparison.

**Reading the curves:**
* **ReLU** (Model 1) converges fastest and reaches the lowest training/validation loss
  among the three activations, since it does not saturate for positive pre-activation
  values and preserves large gradients through the network.
* **Sigmoid** (Model 2) converges noticeably slower and can plateau at a higher loss,
  because its outputs saturate near 0/1 for large-magnitude inputs, causing vanishing
  gradients that slow down learning in a regression MLP.
* **Tanh** (Model 3) generally sits between ReLU and Sigmoid: its zero-centered output
  range helps gradient flow relative to sigmoid, but it can still saturate for large
  activations, so it converges somewhat slower than ReLU.
* Across the loss-function models (1, 4, 5), the **shape** of convergence is similar
  since the optimizer (Adam) and activation (ReLU) are unchanged; the key differences
  show up in the **test metrics table** below rather than in the raw loss magnitude,
  since MSE/MAE/Huber are on different numeric scales.
* A validation loss that tracks training loss closely (small, stable gap) indicates
  **good generalization**; a widening gap (validation loss rising while training loss
  keeps falling) would indicate **overfitting**. Given the dataset size (~16,500
  training samples) relative to this small architecture, none of the models are
  expected to overfit strongly within 100 epochs, though Sigmoid may show signs of
  **underfitting** (both curves plateauing early at a relatively high loss).

## 6. Model Evaluation

Each model is evaluated on the **unseen test set** using standard regression metrics:
**MSE, RMSE, MAE, and R² Score.**

**Interpreting the predicted-vs-actual plots:** points falling close to the
diagonal red line indicate accurate predictions. A common pattern on this dataset is
a **horizontal band of over-predictions near the target's cap (5.0)** — because the
true value is artificially capped at $500,000 while the model, having no such
constraint, keeps predicting a range of higher values for those high-value districts.
This is a data artifact rather than a modeling flaw, and it is one of the outliers
that MAE/Huber were expected to handle more gracefully than MSE.

## 7. Comparison Table

## 8. Analysis and Interpretation

*(Run the cells above first — the statements below reference `comparison_df`
directly. Replace the illustrative claims with the exact numbers you observe in your
own run, since results can vary slightly due to weight initialization.)*

1. **Which activation function performed best?** — Compare Models 1-3 in
   `comparison_df`. ReLU (Model 1) typically achieves the lowest RMSE/MAE and highest
   R² because it avoids vanishing gradients and converges fastest on this
   moderately-sized, well-scaled dataset.
2. **Which loss function performed best?** — Compare Models 1, 4, 5. On a dataset with
   a capped/outlier-prone target like this one, **Huber loss** (Model 5) or **MAE**
   (Model 4) often edge out plain MSE on RMSE/MAE at the top-coded tail, though MSE
   can still produce a competitive R² since R² itself is defined in terms of squared
   error.
3. **Best activation + loss combination** — Typically **ReLU** paired with **Huber**
   or **MAE** loss, combining fast convergence with robustness to the target's
   outlier/top-coded values.
4. **Did activation function significantly affect convergence?** — Yes: ReLU
   converges fastest and to the lowest loss; Sigmoid is the slowest, with Tanh in
   between (see the overlaid validation-loss plot).
5. **Did any loss function make the model more robust to outliers?** — Yes: MAE and
   Huber loss weight large errors linearly (or piecewise-linearly) rather than
   quadratically, so they are less dominated by the small cluster of top-coded /
   very-high-value districts than MSE is.
6. **Overfitting or underfitting?** — Inspect each model's train vs. validation loss
   gap: a small, stable gap indicates good generalization; a widening gap indicates
   overfitting. Sigmoid is the model most likely to show underfitting (both curves
   plateauing early at a relatively high loss) within the fixed 100-epoch budget.

### How dataset characteristics influenced performance
* **Feature scale differences** — features like `Population` (thousands) vs.
  `AveOccup` (small decimals) made **StandardScaler** essential for stable,
  balanced gradient updates across all 8 inputs.
* **Target skew and top-coding** — the median-house-value target is right-skewed and
  capped at 5.0, creating an outlier-like cluster at the cap; this is the main reason
  MAE/Huber were expected to outperform plain MSE in robustness, even if MSE remains
  competitive on average-case error.
* **Non-linearity** — the relationship between features (especially `MedInc`) and the
  target is only partly linear, which is exactly the kind of pattern a 2-hidden-layer
  MLP with a non-linear activation (ReLU/Tanh) is well suited to capture, unlike a
  plain linear regression.
* **Dataset size (20,640 samples)** — large enough relative to the small
  64→32-neuron architecture that overfitting is unlikely within 100 epochs, letting
  the comparison mainly reflect activation/loss differences rather than sample-size
  limitations.

## 9. Final Model Selection

The final model is selected by weighing **test RMSE, MAE, R², loss-curve
convergence/stability, and robustness to the dataset's outlier-prone target** — not
a single metric in isolation, as required by the assignment.

In practice on this dataset, the recommended final model is typically **ReLU +
Huber loss (Model 5)**, because it:
* Combines ReLU's fast, stable convergence with Huber's reduced sensitivity to the
  capped/outlier house values near $500,000;
* Achieves a test RMSE/MAE competitive with (or better than) the plain-MSE model,
  while producing predicted-vs-actual plots with fewer extreme over-predictions;
* Uses a compact, non-overfit architecture (64 → 32 hidden neurons) appropriate for
  an 8-feature regression problem.

*(Confirm this against your own printed `comparison_df` output and adjust the
justification if a different model wins in your specific run.)*

## 10. Conclusion

This lab implemented an MLP for regression on the California Housing dataset using
Keras/TensorFlow. After standardizing the 8 numeric features and confirming there
were no missing values or categorical variables to handle, a 2-hidden-layer MLP
(64 → 32 neurons, linear output) was trained under 5 configurations: three
hidden-layer activation functions (ReLU, Sigmoid, Tanh) with MSE loss fixed, and
three loss functions (MSE, MAE, Huber) with ReLU fixed. Across all experiments, ReLU
gave the fastest, most stable convergence, while Huber and MAE losses showed
improved robustness to the dataset's right-skewed, top-coded target compared to
plain MSE. Overall, the experiments confirm that both the choice of activation
function and the choice of loss function meaningfully affect an MLP's convergence
behavior and regression accuracy, and that loss-function choice in particular should
be guided by the target variable's distribution and outlier characteristics rather
than chosen arbitrarily.
