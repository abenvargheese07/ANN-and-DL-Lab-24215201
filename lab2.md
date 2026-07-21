# Single Artificial Neuron with Backpropagation

A pure Python implementation of a single artificial neuron trained on the logical **OR** gate using continuous Sigmoid activation and Stochastic Gradient Descent (SGD).

Unlike discrete perceptrons, this implementation utilizes differential calculus and the chain rule (backpropagation) to calculate gradients and update internal parameters smoothly over training epochs.

---

## 📌 Key Concepts

1. **Linear Projection (Logit):**  
   Computes the raw score $Z = \sum (W_i \cdot X_i) + B$.
2. **Sigmoid Activation Function:**  
   Squashes the raw score into a continuous probability range $(0, 1)$:  
   $$\sigma(Z) = \frac{1}{1 + e^{-Z}}$$
3. **Loss Function (Mean Squared Error):**  
   Measures prediction variance against the target output:  
   $$L = (Y_{target} - Y_{pred})^2$$
4. **Backpropagation (Chain Rule):**  
   Calculates partial derivatives of loss with respect to weights and bias to perform gradient updates:  
   $$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial Y_{pred}} \cdot \frac{\partial Y_{pred}}{\partial Z} \cdot \frac{\partial Z}{\partial W_i}$$

---

## 🚀 How to Run

No third-party dependencies are required (built strictly using Python standard libraries `random` and `math`).

### 1. Local Execution (Terminal / Mac)

```bash
python3 neuron.py