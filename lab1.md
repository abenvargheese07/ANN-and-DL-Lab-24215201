# Single-Layer Perceptron (Binary Classifier)

A lightweight, pure Python implementation of a Single-Layer Perceptron trained on basic logical operations (AND & OR gates).

This project demonstrates fundamental Machine Learning concepts from scratch—including linear combination, activation functions, and gradient descent—without relying on high-level frameworks like TensorFlow or PyTorch.

---

## 📌 How It Works (In Simple Words)

A Perceptron is the building block of an artificial neural network. It acts like a simple decision-maker:

1. **Inputs ($X$):** The features/data fed into the model.
2. **Weights ($W$):** Numbers that determine how important each input feature is.
3. **Bias ($B$):** An extra number that adjusts the threshold needed to trigger a output.
4. **Weighted Sum ($Z$):** Calculates $Z = (W_1 \cdot X_1 + W_2 \cdot X_2 + \dots) + B$.
5. **Activation Function:** 
   - **Step Function:** Outputs `1` if $Z \ge 0$, otherwise `0`.
   - **Sigmoid Function:** Squashes $Z$ into a smooth decimal value between `0` and `1`.
6. **Training:** The model makes a guess, compares it to the correct answer, and adjusts its weights and bias using **Gradient Descent** until errors reach zero.

---

## 🚀 Setup & Usage

### Prerequisites

Ensure Python 3.x and NumPy are installed:

```bash
pip install numpy
```
---
## 🌍 Real Life Examples

| Example | How the Perceptron Applies |
|---|---|
| **Email Spam Filter (Simple)** | A basic spam filter can treat features like "contains the word *free*" (X₁) and "has a suspicious link" (X₂) as binary inputs. A perceptron learns weights indicating how strongly each feature signals spam, outputting `1` (spam) or `0` (not spam). |
| **Factory Quality Control** | On a production line, sensors check two conditions — e.g., "weight within range" and "color within tolerance". An AND-gate perceptron approves a product (`1`) only when **both** conditions pass, mirroring simple pass/fail inspection logic. |
| **Light Switch Logic (OR Gate)** | A hallway light controlled by two switches at either end is an OR operation — flipping **either** switch turns the light on. The perceptron models this: output is `1` if at least one input is `1`. |

---

## 🔬 Observation

1. **Convergence:** For linearly separable problems (AND, OR), the perceptron converges to zero cumulative loss within a small number of epochs (typically 3–5 iterations with `η = 0.1`).
2. **Weight Evolution:** Weights start at zero and progressively adjust. For the AND gate, both weights grow positive (both inputs must contribute) while the bias becomes negative (raising the threshold so both inputs are needed).
3. **Step Activation:** The Heaviside step function produces hard binary decisions (`0` or `1`) — there is no notion of confidence or probability in the output.
4. **Learning Rate Sensitivity:** A higher `η` leads to faster convergence on these small datasets, but in general can cause oscillation on noisier, larger-scale problems.

---

## 💡 Interpretation

- The Single-Layer Perceptron can only learn **linearly separable** functions. It successfully models AND and OR gates because these truth tables can be separated by a single straight line (decision boundary) in 2D input space.
- It **cannot** learn non-linearly separable functions like XOR — this is the fundamental limitation that motivated the development of multi-layer networks (MLPs) covered in later labs.
- The weight update rule `W += η · (target − predicted) · X` is a special case of gradient descent for the step activation. Each misclassification nudges the decision boundary toward the correct classification.
- The trained bias term acts as a **threshold**: for the AND gate the bias is negative, requiring both inputs to be `1` to overcome it, while for the OR gate the bias is near zero or slightly negative, so a single active input suffices.
