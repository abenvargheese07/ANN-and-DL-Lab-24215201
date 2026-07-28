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
python3 lab2.py

---

## 🌍 Real Life Examples

| Example | How the Sigmoid Neuron Applies |
|---|---|
| **Medical Screening** | A single neuron can estimate the probability of a condition (e.g., diabetes risk) from two features — blood sugar level and BMI. The sigmoid output (`0.82`) gives a **confidence score**, not just a binary yes/no, helping doctors prioritize follow-ups. |
| **Sentiment Analysis (Simple)** | Given two engineered features like "count of positive words" and "count of negative words", a sigmoid neuron outputs a value close to `1` for positive sentiment and close to `0` for negative — acting as a basic opinion classifier. |
| **Loan Default Prediction** | Using debt-to-income ratio and credit score as inputs, the neuron outputs a probability of default (e.g., `0.73`). The bank can set a threshold (say `0.5`) to approve or reject, but also use the raw probability for risk-tiered pricing. |

---

## 🔬 Observation

1. **Smooth Loss Decay:** Unlike the step-function perceptron that jumps to zero loss abruptly, the sigmoid neuron's MSE loss decreases **smoothly and monotonically** over epochs — visible in the logged `Mean Cost` values dropping from ~0.25 toward ~0.001.
2. **Continuous Outputs:** Predictions are continuous decimals (e.g., `0.9817` instead of `1`). The model never outputs an exact `0` or `1`, but gets asymptotically close with more training.
3. **Learning Rate Impact:** With `lr = 0.5`, the OR gate converges in ~600–800 iterations. Lower learning rates (e.g., `0.1`) would require proportionally more iterations but produce smoother weight updates.
4. **Weight Symmetry:** Since both inputs contribute equally to the OR function, the trained weights converge to roughly **equal positive values**, and the bias becomes a moderate negative number — just enough so that a single active input pushes the sigmoid past `0.5`.

---

## 💡 Interpretation

- **Backpropagation** makes learning possible with the smooth sigmoid activation. The chain rule decomposes the total loss gradient into three factors: `∂Loss/∂output × ∂output/∂logit × ∂logit/∂weight`, enabling precise, calculus-based weight updates rather than the crude error-based rule of the step perceptron.
- The **sigmoid activation** converts the neuron from a hard classifier into a **probabilistic model** — the output can be interpreted as the probability of class `1`. This is a major improvement over Lab 1's step function.
- **MSE Loss** works here because it is a simple binary problem. For multi-class or more complex tasks, cross-entropy loss (used in Lab 4) is preferred because it penalizes confident wrong predictions more harshly.
- This single sigmoid neuron is mathematically equivalent to **logistic regression** — the simplest neural network. Stacking multiple such neurons into layers (as in Lab 4's MLP) unlocks the ability to learn non-linear boundaries like XOR.
