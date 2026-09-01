# ANN & Deep Learning — Complete Lab Explanation (Lab 1–7)

> A detailed walkthrough of every lab in the course, covering theory, code, architecture, observations, and how each lab builds upon the previous one.

---

## Table of Contents

1. [Lab 1 — Single-Layer Perceptron (Binary Classifier)](#lab-1--single-layer-perceptron-binary-classifier)
2. [Lab 2 — Single Artificial Neuron with Backpropagation](#lab-2--single-artificial-neuron-with-backpropagation)
3. [Lab 3 — Feedforward Neural Network using PyTorch](#lab-3--feedforward-neural-network-using-pytorch)
4. [Lab 4 — Multi-Class Classification using MLP (Keras)](#lab-4--multi-class-classification-using-mlp-keras)
5. [Lab 5 — CNN for Binary Image Classification (Cats vs. Dogs)](#lab-5--cnn-for-binary-image-classification-cats-vs-dogs)
6. [Lab 6 — Keras MLP for Regression](#lab-6--keras-mlp-for-regression)
7. [Lab 7 — Keras MLP for Multiclass Classification](#lab-7--keras-mlp-for-multiclass-classification)
8. [How the Labs Connect — The Big Picture](#how-the-labs-connect--the-big-picture)

---

---

## Lab 1 — Single-Layer Perceptron (Binary Classifier)

**File:** [`lab_1.py`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL/Lab%201/lab_1.py)

### What This Lab Is About

This is the very first lab and introduces the **simplest possible neural network** — a single-layer perceptron. It is implemented from scratch in pure Python + NumPy (no frameworks like TensorFlow or PyTorch) and trained on basic **logical gates** (AND & OR).

### Core Concepts

| Concept | Explanation |
|---|---|
| **Inputs (X)** | The features/data points fed into the model (e.g., two binary inputs for a logic gate) |
| **Weights (W)** | Numbers that determine how important each input is — the model *learns* these |
| **Bias (B)** | An extra learnable parameter that shifts the decision boundary |
| **Weighted Sum (Z)** | The linear combination: `Z = W₁·X₁ + W₂·X₂ + B` |
| **Activation Function** | The **step function** (Heaviside): outputs `1` if Z ≥ 0, else `0` |
| **Training** | Uses the **Perceptron Learning Rule**, a special case of gradient descent |

### How the Code Works

```python
class SingleLayerPerceptron:
    def __init__(self, input_dim, eta=0.1):
        self.w = np.zeros(input_dim)   # weights start at zero
        self.b = 0.0                    # bias starts at zero
        self.eta = eta                  # learning rate

    def forward(self, feature_vector):
        score = np.dot(self.w, feature_vector) + self.b  # linear combination
        return 1 if score >= 0 else 0                     # step activation

    def fit(self, X_train, y_train, max_epochs=10):
        for epoch in range(max_epochs):
            for sample, target in zip(X_train, y_train):
                y_hat = self.forward(sample)
                delta = target - y_hat
                self.w += self.eta * delta * sample  # weight update
                self.b += self.eta * delta           # bias update
```

**Step-by-step:**

1. **Initialize** all weights and bias to zero.
2. **Forward pass**: compute `Z = W·X + B`, then apply step function to get a prediction (0 or 1).
3. **Compute error**: `delta = target - predicted`.
4. **Update rule**: `W_new = W_old + η × delta × X` and `B_new = B_old + η × delta`.
5. **Repeat** for all samples in each epoch until the error is zero.

### Dataset

The AND gate truth table:

| X₁ | X₂ | AND Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

### Key Observations

- For linearly separable problems (AND, OR), the perceptron converges to **zero error** within 3–5 epochs.
- The trained AND gate has **both weights positive** and a **negative bias** — meaning both inputs must be active to cross the threshold.
- The step function gives **hard binary outputs** — there is no notion of confidence or probability.

### Fundamental Limitation

The single-layer perceptron can **only learn linearly separable functions**. It works for AND and OR, but **fails completely on XOR** because XOR cannot be separated by a single straight line. This limitation is what motivates multi-layer networks in later labs.

### Real-Life Analogy

Think of a simple spam filter: if the email contains "free" (X₁ = 1) **AND** has a suspicious link (X₂ = 1), mark it as spam (output = 1). The perceptron learns the weights for each feature to make this decision.

---

---

## Lab 2 — Single Artificial Neuron with Backpropagation

**File:** [`lab_2.py`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL/Lab%202/lab_2.py)

### What This Lab Is About

This lab upgrades the perceptron from Lab 1 by replacing the **step function** with a **sigmoid activation** and introducing **backpropagation** (the chain rule) for computing gradients. It is still a single neuron, but now it outputs **continuous probabilities** instead of hard 0/1 decisions, and uses calculus-based weight updates.

### What Changed from Lab 1

| Feature | Lab 1 (Perceptron) | Lab 2 (Sigmoid Neuron) |
|---|---|---|
| Activation | Step function (hard 0 or 1) | Sigmoid (smooth value between 0 and 1) |
| Output type | Binary decision | Probability / confidence score |
| Loss function | None explicit (just error counting) | Mean Squared Error (MSE) |
| Weight update | Perceptron rule (heuristic) | **Backpropagation** (chain rule, calculus-based) |
| Framework | NumPy | Pure Python (`math`, `random`) |

### Core Concepts

**Sigmoid Activation:**

$$\sigma(Z) = \frac{1}{1 + e^{-Z}}$$

- Squashes any real number into the range (0, 1).
- Output can be interpreted as a **probability** of class 1.
- Unlike the step function, it is **differentiable** — critical for backpropagation.

**Backpropagation (Chain Rule):**

The gradient of the loss with respect to each weight is decomposed using the chain rule:

$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial Y_{pred}} \times \frac{\partial Y_{pred}}{\partial Z} \times \frac{\partial Z}{\partial W_i}$$

Where:
- `∂L/∂Y_pred = -2(target - predicted)` — how loss changes with output
- `∂Y_pred/∂Z = σ(Z) × (1 - σ(Z))` — sigmoid derivative
- `∂Z/∂W_i = X_i` — how the logit changes with each weight

### How the Code Works

```python
def optimize_sample(self, x_vector, ground_truth):
    y_hat = self.predict(x_vector)              # forward pass (sigmoid output)
    residual = ground_truth - y_hat             # error
    cost = residual ** 2                        # MSE loss

    # Backward pass — chain rule
    d_cost_d_out = -2 * residual                # ∂L/∂ŷ
    d_out_d_logit = y_hat * (1 - y_hat)         # sigmoid derivative
    delta_logit = d_cost_d_out * d_out_d_logit  # combined gradient

    # Update each weight
    for idx in range(len(self.params)):
        grad_w = delta_logit * x_vector[idx]    # ∂Z/∂W_i
        self.params[idx] -= self.lr * grad_w    # SGD update

    self.offset -= self.lr * delta_logit        # bias update
```

### Dataset

The OR gate truth table, trained over 1000 iterations with learning rate 0.5:

| X₁ | X₂ | OR Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

### Key Observations

- Unlike the step perceptron, the MSE loss decreases **smoothly and monotonically** over epochs (no abrupt jumps).
- Predictions are continuous decimals (e.g., `0.9817` instead of exactly `1`). The model never outputs an exact 0 or 1 — it gets asymptotically close.
- Both trained weights converge to roughly **equal positive values** (since both inputs contribute equally to OR), and the bias becomes a moderate negative number.
- This single sigmoid neuron is mathematically equivalent to **logistic regression**.

### Why This Matters

Backpropagation is **the** core algorithm of deep learning. Every modern neural network — no matter how large — uses the same chain rule principle introduced here. Lab 2 shows it in its simplest possible form: one neuron, two inputs, manual gradient computation.

---

---

## Lab 3 — Feedforward Neural Network using PyTorch

**File:** [`lab3.py`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL/Lab%203/lab3.py)

### What This Lab Is About

This lab bridges the gap between the **from-scratch implementations** (Labs 1 & 2) and **high-level frameworks**. It introduces **PyTorch** and builds a simple feedforward neural network with **one hidden layer** using the `nn.Module` abstraction, demonstrating the standard PyTorch training loop.

### Architecture

```
Input (10) → Linear(10, 32) → ReLU → Linear(32, 1) → Output
```

| Layer | Type | Input → Output | Activation |
|---|---|---|---|
| `fc1` | Linear | 10 → 32 | ReLU |
| `fc2` | Linear | 32 → 1 | None (linear) |

**Total trainable parameters:** (10×32 + 32) + (32×1 + 1) = **385**

### New Concept: ReLU Activation

$$\text{ReLU}(x) = \max(0, x)$$

- Outputs zero for negative inputs, passes positive inputs through unchanged.
- **No vanishing gradient** for positive values (unlike sigmoid, which saturates).
- The most widely used activation function in modern deep learning.
- Without a non-linear activation between layers, stacking linear layers would collapse into a single linear transformation — making the hidden layer useless.

### The PyTorch Training Loop (5 Steps)

```python
# 1. Clear old gradients
optimizer.zero_grad()

# 2. Forward pass: compute predictions
outputs = model(x_dummy)

# 3. Compute loss
loss = criterion(outputs, y_dummy)

# 4. Backward pass: compute gradients via autograd
loss.backward()

# 5. Update weights using optimizer
optimizer.step()
```

This is the **canonical PyTorch training pattern** — memorize it, every PyTorch project uses it.

### Key Concepts Introduced

| Concept | Explanation |
|---|---|
| **`nn.Module`** | PyTorch's base class for neural networks; you define layers in `__init__` and the computation graph in `forward` |
| **Autograd** | PyTorch automatically computes all gradients via `loss.backward()` — no manual chain rule needed |
| **Adam Optimizer** | Adaptive optimizer that adjusts learning rates per-parameter; faster convergence than vanilla SGD |
| **MSE Loss** | Appropriate for regression tasks (predicting continuous values) |

### Key Observations

- The script runs only **one training step** on random data — the focus is on demonstrating the **mechanics** of the training loop, not achieving convergence.
- After `loss.backward()`, every parameter has a `.grad` attribute containing its computed gradient.
- Without `optimizer.zero_grad()`, gradients would **accumulate** across steps (PyTorch's default behavior), leading to incorrect updates — this is a very common source of bugs.

### How This Differs from Labs 1 & 2

In Labs 1 & 2, you manually computed every gradient using explicit math. PyTorch's **autograd** system automates this entirely — you just define the forward computation, and PyTorch handles the backward pass. This is what makes deep learning on complex architectures practical.

---

---

## Lab 4 — Multi-Class Classification using MLP (Keras)

**File:** [`lab4.py`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL/Lab%204/lab4.py)

### What This Lab Is About

This lab builds a **deep MLP** using TensorFlow/Keras for **multi-class classification** (5 classes) on a synthetic dataset. It introduces several key regularization and training techniques used in modern deep learning: **Batch Normalization**, **Dropout**, **Early Stopping**, and **Learning Rate Scheduling**.

### Dataset

Generated synthetically using scikit-learn:

| Parameter | Value |
|---|---|
| Samples | 5,000 |
| Features | 20 (15 informative, 5 redundant) |
| Classes | 5 |
| Split | 64% train / 16% validation / 20% test |
| Preprocessing | StandardScaler (fit on train only) |

### Architecture

```
Input (20) → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
           → Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
           → Dense(32, ReLU)
           → Dense(5, Softmax)
```

### New Concepts Introduced

**1. Softmax Activation (Output Layer)**

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

- Converts raw logits into a **probability distribution** across all classes.
- All outputs sum to 1.0, allowing the model to express confidence (e.g., "70% class A, 25% class B, 5% class C").

**2. Batch Normalization**
- Normalizes the inputs to each layer (zero mean, unit variance) during training.
- **Stabilizes and accelerates training** by reducing internal covariate shift.
- Without it, the same architecture often shows erratic loss spikes during early epochs.

**3. Dropout (0.3)**
- During training, randomly **deactivates 30%** of neurons in each dropout layer.
- Forces the network to learn **redundant, robust representations** rather than relying on any single neuron.
- Training accuracy is intentionally lower than validation accuracy (because dropout is active during training), but test accuracy closely matches validation — confirming that dropout effectively prevents overfitting.

**4. Early Stopping (patience = 10)**
- Monitors validation loss each epoch.
- If validation loss doesn't improve for 10 consecutive epochs, training halts.
- `restore_best_weights=True` ensures the final model uses the weights from the epoch with the **lowest** validation loss, not the last epoch.

**5. ReduceLROnPlateau (factor = 0.5, patience = 5)**
- If validation loss plateaus for 5 epochs, the learning rate is halved.
- Allows finer weight adjustments in later training stages.
- Typically fires 1–2 times during training, squeezing out an extra 1–2% accuracy.

**6. Sparse Categorical Crossentropy**
- The appropriate loss function when class labels are integers (0, 1, 2, 3, 4) rather than one-hot encoded.
- Penalizes confident wrong predictions much more harshly than MSE (used in Lab 2).

### Why This Is a Big Step Up

| Previous Labs | Lab 4 |
|---|---|
| Single neuron / one hidden layer | 3 hidden layers (128 → 64 → 32) |
| Binary classification (2 classes) | Multi-class classification (5 classes) |
| No regularization | BatchNorm + Dropout + Early Stopping + LR scheduling |
| Manual training loop (PyTorch) or from-scratch | Keras `.fit()` handles everything |
| Toy data (4 samples) | 5,000 samples with realistic feature structure |

### Key Observations

- The model typically stops training around epoch 25–40 (well before the 100-epoch maximum) thanks to early stopping.
- Test accuracy ranges from ~85–92% on the synthetic 5-class dataset.
- The combination of BatchNorm + Dropout + Early Stopping + LR scheduling represents a **modern deep learning regularization stack** — each technique addresses a different failure mode.

---

---

## Lab 5 — CNN for Binary Image Classification (Cats vs. Dogs)

**File:** [`lab5.ipynb`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL/Lab%205/lab5.ipynb)

### What This Lab Is About

This is the first lab to work with **images**. It builds a **Convolutional Neural Network (CNN)** using TensorFlow/Keras for binary classification (cats vs. dogs), covering image preprocessing, data augmentation, CNN architecture design, and interpretation of predictions.

### Dataset

| Property | Value |
|---|---|
| Source | "Cats and Dogs Filtered" (Google-hosted TensorFlow tutorial dataset) |
| Training images | 2,000 (1,000 cats + 1,000 dogs) |
| Validation images | 1,000 (500 cats + 500 dogs) |
| Image size | Resized to 150×150×3 (RGB) |
| Classes | 2 (cat = 0, dog = 1) |
| Balance | Perfectly balanced |

### Why CNNs Instead of MLPs for Images?

An MLP would treat each pixel independently — a 150×150×3 image has 67,500 input features, which is impractical and ignores spatial structure. CNNs exploit the fact that nearby pixels are related by using **convolutional filters** that slide across the image, learning **local patterns** (edges, textures, shapes) that are translation-invariant.

### Architecture

```
Input (150×150×3)
  → Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)     # learns edges, simple textures
  → Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)     # learns texture combinations
  → Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)    # learns parts (ears, eyes)
  → Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)    # learns high-level structure
  → Flatten
  → Dropout(0.5)
  → Dense(512, ReLU)
  → Dense(1, Sigmoid)                             # binary output
```

### New Concepts Introduced

**1. Convolutional Layers (Conv2D)**
- Apply small learnable filters (3×3) that slide across the image.
- Each filter detects a specific pattern (edges, curves, textures).
- **Parameter sharing**: the same filter weights are used across the entire image, drastically reducing parameters compared to a fully connected layer.

**2. MaxPooling**
- Reduces spatial dimensions by taking the maximum value in each 2×2 window.
- Makes the representation **more compact** and provides mild translation invariance.
- After 4 pooling operations, the 150×150 spatial dimension shrinks to approximately 7×7.

**3. Data Augmentation**
- Randomly transforms training images each epoch: flips, rotations, zooms, contrast changes.
- Artificially expands the effective training data.
- **Critical for small datasets** — without augmentation on 2,000 images, the model quickly memorizes the training set (overfitting).
- **Never applied to validation/test data** — evaluation must reflect performance on real, unmodified images.

**4. Binary Cross-Entropy Loss**
- The correct loss function for binary classification with a sigmoid output.
- Penalizes confident wrong predictions exponentially harder than near-threshold mistakes.

### Key Experiment: With vs. Without Augmentation

The notebook trains **two identical CNNs** — one with and one without data augmentation:

| Metric | Without Augmentation | With Augmentation |
|---|---|---|
| Training vs. Validation gap | **Large** (overfitting) | **Small** (good generalization) |
| Training accuracy | Very high (memorizing) | Lower but honest |
| Validation accuracy | Lower | Higher |

This controlled experiment directly demonstrates augmentation's **regularizing effect**.

### Key Observations

- The CNN design pattern of **doubling filters while halving spatial resolution** (32→64→128→128 with pooling) is a standard approach used in VGG, ResNet, and most modern architectures.
- Misclassifications mostly occur on **visually ambiguous images** — unusual poses, partial occlusion, or breeds that look like the other species.
- Dropout(0.5) before the dense layer further reduces overfitting by preventing co-adaptation of neurons in the classification head.

---

---

## Lab 6 — Keras MLP for Regression

**File:** [`Lab6.ipynb`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL/Lab%206/Lab6.ipynb)

### What This Lab Is About

This lab returns to MLPs but tackles a **regression** problem instead of classification — predicting **California house prices** from 8 features. It compares three activation functions and three loss functions across 5 models.

### Dataset

| Property | Value |
|---|---|
| Source | California Housing (scikit-learn, 1990 U.S. Census) |
| Samples | 20,640 |
| Features | 8 numeric (MedInc, HouseAge, AveRooms, etc.) |
| Target | MedHouseVal (median house value in $100,000s, capped at 5.0) |
| Split | 80% train / 20% test, with 20% of train as validation |
| Preprocessing | StandardScaler on features (target left unscaled for interpretability) |

### Architecture

```
Input (8) → Dense(64, activation=varied) → Dense(32, activation=varied) → Dense(1, linear)
```

The output layer uses **linear activation** (no squashing) because regression targets can be any real number.

### Two Experiments

**Experiment A — Activation Functions** (loss fixed to MSE):

| Model | Activation | Optimizer | Loss |
|---|---|---|---|
| Model 1 | **ReLU** | Adam | MSE |
| Model 2 | **Sigmoid** | Adam | MSE |
| Model 3 | **Tanh** | Adam | MSE |

**Experiment B — Loss Functions** (activation fixed to ReLU):

| Model | Activation | Optimizer | Loss |
|---|---|---|---|
| Model 1 | ReLU | Adam | **MSE** |
| Model 4 | ReLU | Adam | **MAE** |
| Model 5 | ReLU | Adam | **Huber** |

### New Concepts: Regression Losses

| Loss Function | Formula | Behavior |
|---|---|---|
| **MSE** | (y - ŷ)² | Squares errors → very sensitive to outliers |
| **MAE** | \|y - ŷ\| | Linear penalty → robust to outliers |
| **Huber** | MSE for small errors, MAE for large errors | Best of both worlds |

**Why this matters for this dataset:** The target is **capped at 5.0** ($500K) — a cluster of artificially top-coded values acts like outliers. MSE is heavily influenced by these, while MAE and Huber handle them more gracefully.

### Key Differences from Classification Labs

| Classification (Labs 4, 7) | Regression (Lab 6) |
|---|---|
| Output: softmax (probability distribution) | Output: **linear** (unbounded real number) |
| Loss: cross-entropy | Loss: MSE / MAE / Huber |
| Metrics: accuracy, precision, recall, F1 | Metrics: **MAE, RMSE, R²** |
| Evaluation: confusion matrix | Evaluation: **predicted-vs-actual scatter plot** |

### Key Findings

- **ReLU** converges fastest; **Sigmoid** is slowest (vanishing gradients are especially problematic in regression because saturated neurons can't produce the full range of target values).
- **Huber loss** or **MAE** typically edge out MSE at the capped tail of the target, producing fewer extreme over-predictions.
- Best combo: **ReLU + Huber** — fast convergence with outlier robustness.
- The predicted-vs-actual plots show a characteristic **horizontal band near 5.0** — the model predicts a range of values for districts where the true value is capped, which is a data artifact, not a model flaw.

### Evaluation Metrics Explained

| Metric | Meaning | Ideal Value |
|---|---|---|
| **MAE** | Average absolute error in $100,000s | Lower is better |
| **RMSE** | Square root of average squared error (penalizes large errors more) | Lower is better |
| **R²** | Proportion of variance explained by the model (1.0 = perfect) | Closer to 1.0 |

---

---

## Lab 7 — Keras MLP for Multiclass Classification

**File:** [`Lab7.ipynb`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL/Lab%207/Lab7.ipynb)

### What This Lab Is About

This lab builds an MLP for **multiclass classification** (3 classes) using the UCI Wine dataset, comparing three activation functions and three optimizers across 5 models. It is structurally similar to Lab 6 but focuses on **classification** rather than regression, and varies **optimizers** instead of loss functions.

### Dataset

| Property | Value |
|---|---|
| Source | Wine dataset (scikit-learn, UCI ML Repository) |
| Samples | 178 |
| Features | 13 numeric (alcohol, malic acid, ash, flavanoids, etc.) |
| Target | Cultivar (3 classes: 0, 1, 2) |
| Class balance | 59 / 71 / 48 (reasonably balanced) |
| Split | 80/20 stratified, with 20% of train as validation |
| Feature preprocessing | StandardScaler |
| Target preprocessing | One-hot encoding (for softmax + categorical crossentropy) |

### Architecture

```
Input (13) → Dense(32, activation=varied) → Dense(16, activation=varied) → Dense(3, Softmax)
```

Batch size = 16, epochs = 100, optimizer = varied.

### Two Experiments

**Experiment A — Activation Functions** (optimizer fixed to Adam):

| Model | Activation | Optimizer |
|---|---|---|
| Model 1 | **ReLU** | Adam |
| Model 2 | **Sigmoid** | Adam |
| Model 3 | **Tanh** | Adam |

**Experiment B — Optimizers** (activation fixed to ReLU):

| Model | Activation | Optimizer |
|---|---|---|
| Model 1 | ReLU | **Adam** |
| Model 4 | ReLU | **SGD** |
| Model 5 | ReLU | **RMSprop** |

Model 1 (ReLU + Adam) is shared between both experiments.

### New Concepts: Optimizer Comparison

| Optimizer | How It Works | Convergence |
|---|---|---|
| **SGD** | Fixed learning rate, no momentum | Slowest — may not converge in 100 epochs |
| **Adam** | Adaptive LR per-parameter + momentum | Fast, smooth convergence |
| **RMSprop** | Adaptive LR per-parameter (no momentum) | Fast, similar to Adam |

**Why SGD is slower:** It uses the same learning rate for all parameters. If some features have larger gradients than others (common with varied feature scales), SGD either updates too aggressively on some or too slowly on others. Adam and RMSprop adapt per-parameter, solving this.

### Activation Function Comparison (Revisited)

| Activation | Pros | Cons |
|---|---|---|
| **ReLU** | No vanishing gradient for positive inputs; fast convergence | "Dead neurons" if inputs are always negative |
| **Sigmoid** | Outputs in (0,1); smooth | Vanishing gradients for large inputs; not zero-centered |
| **Tanh** | Zero-centered (better than sigmoid) | Still saturates for large inputs |

### Key Findings

- **ReLU + Adam** typically gives the best test accuracy and F1-score.
- **SGD** without momentum is the most likely to show a still-improving curve at epoch 100 (under-convergence, not a modeling problem — just needs more epochs).
- **Misclassifications** mostly occur between class 1 and class 2, whose chemical profiles overlap.
- The confusion matrices show that class 0 is almost always classified correctly (its chemical features are very distinct).

### Evaluation

Each model is evaluated using:
- **Confusion Matrix** — shows where misclassifications happen (which classes are confused)
- **Macro-Averaged Precision, Recall, F1** — treats each class equally, appropriate when classes are roughly balanced
- **Training vs. Validation Curves** — to diagnose overfitting/underfitting

---

---

## How the Labs Connect — The Big Picture

```
Lab 1                Lab 2                Lab 3              Lab 4
Single-Layer     →   Sigmoid Neuron   →   PyTorch FNN    →   Deep MLP
Perceptron           + Backprop           + Autograd          + Regularization
(Step function,      (Smooth output,      (Framework intro,   (BatchNorm, Dropout,
 AND/OR gates)        chain rule,          ReLU, Adam)         5-class, Keras)
                      MSE loss)

                                                                    ↓

Lab 7                Lab 6                Lab 5
MLP Multiclass   ←   MLP Regression   ←   CNN for Images
Classification       (House prices,       (Cats vs. Dogs,
(Wine, optimizers    3 loss functions      Conv2D, MaxPool,
 vs activations)      compared)            data augmentation)
```

### Progression of Complexity

| Lab | Model | Framework | Data | Key New Idea |
|---|---|---|---|---|
| **1** | 1 neuron, step | NumPy | 4 samples (AND) | Perceptron learning rule |
| **2** | 1 neuron, sigmoid | Pure Python | 4 samples (OR) | Backpropagation (chain rule) |
| **3** | 1 hidden layer | PyTorch | Random data | Framework training loop, autograd |
| **4** | 3 hidden layers | Keras | 5,000 synthetic | Regularization stack (BN, Dropout, EarlyStopping) |
| **5** | 4-block CNN | Keras | 3,000 images | Convolutions, pooling, data augmentation |
| **6** | 2 hidden layers | Keras | 20,640 real samples | Regression, 3 loss functions compared |
| **7** | 2 hidden layers | Keras | 178 real samples | 3 activations × 3 optimizers comparison |

### Key Takeaways Across All Labs

1. **Activation functions matter**: ReLU consistently outperforms Sigmoid and Tanh for hidden layers because it avoids vanishing gradients.
2. **Optimizers matter**: Adaptive optimizers (Adam, RMSprop) converge faster and more reliably than vanilla SGD.
3. **Loss function should match the task**: MSE for regression, cross-entropy for classification, and consider Huber/MAE when outliers are present.
4. **Regularization prevents overfitting**: Dropout, BatchNorm, Early Stopping, Data Augmentation — each addresses a different failure mode.
5. **Feature scaling is essential**: StandardScaler (or equivalent) is critical for gradient-based optimizers to converge properly.
6. **Architecture should match the data**: MLPs for tabular data, CNNs for images. Output activation should match the task (linear for regression, softmax for multi-class, sigmoid for binary).
7. **Fair comparisons require controlled experiments**: Change only one variable at a time (activation OR optimizer, not both) while holding everything else constant.

---

> **Source files:** All lab code and documentation can be found in the [`ANN DL`](file:///Users/abenvargheese/Documents/Coding/5th%20Sem/ANN%20DL) directory.
