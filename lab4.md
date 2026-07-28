# Lab 4 – Multi-Class Classification using Multi-Layer Perceptron (MLP)

## Objective

Build and train a deep neural network (Multi-Layer Perceptron) using TensorFlow/Keras for **multi-class classification** on a synthetic dataset. The lab demonstrates key deep learning techniques including batch normalization, dropout regularization, learning rate scheduling, and early stopping.

## Dataset

A synthetic classification dataset is generated using `sklearn.datasets.make_classification`:

| Parameter         | Value |
|-------------------|-------|
| Samples           | 5000  |
| Features          | 20    |
| Informative       | 15    |
| Redundant         | 5     |
| Classes           | 5     |

### Data Splits

| Split      | Proportion |
|------------|------------|
| Training   | 64%        |
| Validation | 16%        |
| Test       | 20%        |

All splits are **stratified** to maintain class balance. Features are standardized using `StandardScaler`.

## Model Architecture

A Sequential MLP with the following layers:

```
Input (20) → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
           → Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
           → Dense(32, ReLU)
           → Dense(5, Softmax)
```

## Training Configuration

| Hyperparameter    | Value                            |
|-------------------|----------------------------------|
| Optimizer         | Adam (lr = 0.001)                |
| Loss Function     | Sparse Categorical Crossentropy  |
| Batch Size        | 64                               |
| Max Epochs        | 100                              |
| Early Stopping    | patience = 10, restore best wts  |
| LR Reduction      | factor = 0.5, patience = 5       |

## Key Techniques Demonstrated

- **Batch Normalization** – Stabilizes and accelerates training by normalizing layer inputs.
- **Dropout (0.3)** – Prevents overfitting by randomly deactivating 30% of neurons during training.
- **Early Stopping** – Halts training when validation loss stops improving (patience = 10).
- **ReduceLROnPlateau** – Halves the learning rate when validation loss plateaus (patience = 5).
- **Stratified Splitting** – Ensures consistent class distribution across train/val/test sets.

## How to Run

```bash
pip install numpy tensorflow scikit-learn
python lab4.py
```

## Output

- Model summary (layer shapes and parameter counts)
- Training/validation loss and accuracy per epoch
- Final test loss and accuracy
- Sample predictions vs actual labels for the first 5 test samples

## Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| `numpy`        | Numerical operations             |
| `tensorflow`   | Neural network framework         |
| `scikit-learn` | Dataset generation & preprocessing |

---

## 🌍 Real Life Examples

| Example | How the MLP Applies |
|---|---|
| **Disease Diagnosis** | Given patient data (blood pressure, glucose, cholesterol, etc.), an MLP classifies the condition into one of several diseases (diabetes, hypertension, anaemia, healthy, etc.). The softmax output gives a probability distribution across all classes, helping doctors see not just the top diagnosis but also runner-up possibilities. |
| **Product Categorization (E-commerce)** | An online marketplace uses product attributes (weight, dimensions, price range, keyword embeddings) to automatically sort listings into categories like Electronics, Clothing, Home, Sports, and Books — exactly the kind of 5-class problem modeled here. |
| **Handwritten Digit Recognition** | The classic MNIST task classifies handwritten digits (0–9) using pixel intensity features. While CNNs are preferred, an MLP with BatchNorm and Dropout (as in this lab) achieves competitive accuracy and demonstrates the core deep learning pipeline. |

---

## 🔬 Observation

1. **Early Stopping Triggers:** The model typically stops well before the 100-epoch maximum (often around epoch 25–40), indicating that validation loss plateaus early. The `restore_best_weights` option ensures the final model uses the epoch with the lowest validation loss, not the last epoch.
2. **BatchNormalization Effect:** Adding BatchNorm layers between Dense layers noticeably stabilizes training — without it, the same architecture often shows erratic validation loss spikes during early epochs.
3. **Dropout Regularization:** Training accuracy is consistently **lower** than validation accuracy during training (because dropout is active), but test accuracy closely matches validation accuracy (~85–90%), confirming that dropout effectively prevents overfitting.
4. **Learning Rate Reduction:** The `ReduceLROnPlateau` callback typically halves the learning rate 1–2 times during training, allowing finer weight adjustments in later epochs and squeezing out an additional 1–2% accuracy.
5. **Test Accuracy:** On the synthetic 5-class dataset with 5 redundant features, the model achieves approximately **85–92% test accuracy**, depending on the random initialization.

---

## 💡 Interpretation

- The **Multi-Layer Perceptron** overcomes the fundamental limitation of single-layer perceptrons (Labs 1 & 2) — it can learn **non-linear decision boundaries** by stacking multiple layers with non-linear activations (ReLU). This is what makes it capable of handling complex, multi-class problems.
- **Softmax activation** in the output layer converts raw logits into a proper probability distribution across all 5 classes, enabling the model to express uncertainty (e.g., "70% class A, 25% class B, 5% class C").
- **Sparse Categorical Crossentropy** is the appropriate loss function when class labels are integers (0, 1, 2, 3, 4) rather than one-hot encoded vectors. It penalizes confident wrong predictions much more heavily than MSE (used in Lab 2).
- The combination of **BatchNorm + Dropout + Early Stopping + LR scheduling** represents a modern deep learning regularization stack. Each technique addresses a different failure mode: BatchNorm handles internal covariate shift, Dropout prevents co-adaptation of neurons, Early Stopping prevents overtraining, and LR scheduling enables fine-grained convergence.
- **Stratified splitting** is critical for multi-class problems to ensure each class is proportionally represented in train/val/test sets — without it, minority classes could be underrepresented in the test set, leading to misleading accuracy metrics.
