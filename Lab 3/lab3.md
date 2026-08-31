# Lab 3 – Feedforward Neural Network using PyTorch

## Objective

Build a simple **Feedforward Neural Network** (single hidden layer) using PyTorch and demonstrate the core deep learning training loop: forward pass → loss computation → backpropagation → weight update. This lab introduces the PyTorch framework and its `nn.Module` abstraction.

---

## Model Architecture

```
Input (10) → Linear(10, 32) → ReLU → Linear(32, 1) → Output
```

| Layer   | Type       | Input Dim | Output Dim | Activation |
|---------|------------|-----------|------------|------------|
| `fc1`   | Linear     | 10        | 32         | ReLU       |
| `fc2`   | Linear     | 32        | 1          | None       |

Total trainable parameters: `(10×32 + 32) + (32×1 + 1)` = **385**

---

## Training Configuration

| Component       | Value / Choice                     |
|-----------------|------------------------------------|
| Framework       | PyTorch (`torch.nn`)               |
| Loss Function   | Mean Squared Error (`nn.MSELoss`)  |
| Optimizer       | Adam (lr = 0.001)                  |
| Batch Size      | 64                                 |
| Input Data      | Random tensor (`torch.randn`)      |
| Training Steps  | 1 (single forward + backward pass) |

---

## Training Pipeline (Step-by-Step)

1. **`optimizer.zero_grad()`** — Clear accumulated gradients from the previous step.
2. **`model(x_dummy)`** — Forward pass through `fc1 → ReLU → fc2`.
3. **`criterion(outputs, y_dummy)`** — Compute MSE loss between predictions and targets.
4. **`loss.backward()`** — Backpropagate to compute gradients for all parameters.
5. **`optimizer.step()`** — Update weights using the Adam optimizer.

---

## Key Concepts Demonstrated

- **`nn.Module` Subclassing** — Defines a reusable network class with `__init__` (layers) and `forward` (computation graph).
- **ReLU Activation** — Introduces non-linearity: `f(x) = max(0, x)`. Without it, stacking linear layers would collapse into a single linear transformation.
- **Autograd (Automatic Differentiation)** — PyTorch's `loss.backward()` automatically computes all gradients via the computational graph — no manual derivative calculations needed.
- **Adam Optimizer** — Adaptive learning rate optimizer that combines momentum and RMSProp, generally converging faster than vanilla SGD.

---

## How to Run

```bash
pip install torch
python lab3.py
```

---

## Output

The script performs a single training step on random data. No printed output by default — it demonstrates the **mechanics** of the PyTorch training loop rather than convergence to a solution.

---

## Dependencies

| Package   | Purpose                                   |
|-----------|-------------------------------------------|
| `torch`   | Deep learning framework (tensors, autograd, nn) |

---

## 🌍 Real Life Examples

| Example | How the Feedforward NN Applies |
|---|---|
| **House Price Prediction** | Given 10 features (area, bedrooms, location score, age, etc.), the network outputs a single continuous value — the predicted price. This is exactly the architecture used here: 10 inputs → hidden layer → 1 output with MSE loss. |
| **Energy Consumption Forecasting** | Power companies predict hourly energy demand from features like temperature, humidity, time of day, and historical usage. A feedforward network with a single hidden layer serves as a fast, lightweight baseline model. |
| **Stock Return Estimation** | Given financial indicators (P/E ratio, volume, moving averages, volatility, etc.) as a 10-dimensional input vector, the network estimates a continuous return value. The MSE loss penalizes large prediction errors proportionally. |

---

## 🔬 Observation

1. **Single-Step Training:** The script performs only **one** gradient update. In a real scenario, this loop would be repeated for hundreds/thousands of epochs over the dataset. After a single step, the loss drops only marginally from its initial value.
2. **Random Data:** Since both inputs and targets are random (`torch.randn`), the model cannot learn a meaningful mapping. This is intentional — the lab focuses on demonstrating the **training mechanics**, not achieving convergence.
3. **Gradient Flow:** After `loss.backward()`, every parameter in the model (weights and biases of both layers) has a `.grad` attribute populated with the computed gradient. These gradients are what `optimizer.step()` uses to update weights.
4. **Zero Grad Necessity:** Without `optimizer.zero_grad()`, gradients would **accumulate** across steps (PyTorch default behavior), leading to incorrect updates. This is a common source of bugs in PyTorch training loops.

---

## 💡 Interpretation

- This lab bridges the gap between the **from-scratch implementations** (Labs 1 & 2) and the **high-level Keras API** (Lab 4). PyTorch gives explicit control over the training loop while automating gradient computation via autograd.
- The **single hidden layer with ReLU** is the simplest non-linear neural network. It can theoretically approximate any continuous function (Universal Approximation Theorem), though deeper networks (Lab 4) are more practical for complex tasks.
- **MSE Loss** is appropriate here because the task is **regression** (predicting a continuous output). For classification, cross-entropy loss would be used instead (as in Lab 4).
- The **Adam optimizer** adapts the learning rate per-parameter using first and second moment estimates of the gradient. This makes it robust to different feature scales and generally requires less hyperparameter tuning than vanilla SGD (used implicitly in Labs 1 & 2).
- Compared to Lab 4's Keras approach (`.fit()` handles everything), PyTorch requires manual loop management (`zero_grad → forward → loss → backward → step`) — this verbosity is by design, offering full transparency and flexibility for research and custom training logic.
