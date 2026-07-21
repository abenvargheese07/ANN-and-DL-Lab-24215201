import random
import math


class SigmoidPerceptron:
    def __init__(self, feature_count, lr=0.1):
        # Initialize parameter vector and bias term with random values
        self.params = [random.uniform(-1, 1) for _ in range(feature_count)]
        self.offset = random.uniform(-1, 1)
        self.lr = lr

    def _compute_logit(self, x_vector):
        # Linear projection: net_input = sum(w_i * x_i) + b
        net_input = sum(w * x for w, x in zip(self.params, x_vector)) + self.offset
        return net_input

    def _sigmoid(self, logit):
        # Logistic activation
        return 1 / (1 + math.exp(-logit))

    def _sigmoid_grad(self, output):
        # Local derivative wrt pre-activation logit
        return output * (1 - output)

    def predict(self, x_vector):
        # Forward inference pass
        logit = self._compute_logit(x_vector)
        y_hat = self._sigmoid(logit)
        return y_hat

    def optimize_sample(self, x_vector, ground_truth):
        # Forward pass
        y_hat = self.predict(x_vector)

        # Loss evaluation (MSE single instance)
        residual = ground_truth - y_hat
        cost = residual ** 2

        # Backward pass gradient derivation (Chain Rule)
        d_cost_d_out = -2 * residual
        d_out_d_logit = self._sigmoid_grad(y_hat)
        delta_logit = d_cost_d_out * d_out_d_logit

        # Parameter updates via Stochastic Gradient Descent
        for idx in range(len(self.params)):
            grad_w = delta_logit * x_vector[idx]
            self.params[idx] -= self.lr * grad_w

        grad_b = delta_logit * 1
        self.offset -= self.lr * grad_b

        return cost

    def fit(self, dataset, max_iterations=1000, display_log=True):
        for current_iter in range(max_iterations):
            epoch_loss = 0
            for x_vector, ground_truth in dataset:
                epoch_loss += self.optimize_sample(x_vector, ground_truth)
            mean_loss = epoch_loss / len(dataset)

            if display_log and current_iter % 100 == 0:
                print(f"Iteration {current_iter:4d} | Mean Cost: {mean_loss:.6f}")


if __name__ == "__main__":
    # Training dataset for logical OR function
    dataset = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
    ]

    model = SigmoidPerceptron(feature_count=2, lr=0.5)

    print("Fitting model to logical OR gate dataset...\n")
    model.fit(dataset, max_iterations=1000)

    print("\nOptimized Parameters:", model.params)
    print("Optimized Offset (Bias):", model.offset)

    print("\nEvaluation:")
    for x_vector, ground_truth in dataset:
        y_hat = model.predict(x_vector)
        print(f"Features: {x_vector} -> Prediction: {y_hat:.4f} | Target: {ground_truth}")