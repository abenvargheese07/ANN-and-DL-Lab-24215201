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