import numpy as np

class SingleLayerPerceptron:
    def __init__(self, input_dim, eta=0.1):
        # Initialize weight vector and offset parameter
        self.w = np.zeros(input_dim)
        self.b = 0.0
        self.eta = eta

    def _activate(self, score):
        # Heaviside step activation
        return 1 if score >= 0 else 0

    def forward(self, feature_vector):
        # Linear combination: z = w^T x + b
        score = np.dot(self.w, feature_vector) + self.b
        return self._activate(score)

    def fit(self, X_train, y_train, max_epochs=10):
        for current_epoch in range(max_epochs):
            cumulative_error = 0
            for sample, target in zip(X_train, y_train):
                y_hat = self.forward(sample)
                delta = target - y_hat
                cumulative_error += abs(delta)

                # Weight update rule
                self.w += self.eta * delta * sample
                self.b += self.eta * delta

            print(f"Iteration {current_epoch + 1}: W = {self.w}, "
                  f"B = {self.b:.2f}, Cumulative Loss = {cumulative_error}")

if __name__ == "__main__":
    # Input dataset for logical AND gate
    X_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_targets = np.array([0, 0, 0, 1])

    model = SingleLayerPerceptron(input_dim=2, eta=0.1)

    print("Fitting model to dataset...\n")
    model.fit(X_data, y_targets, max_epochs=10)

    print("\nTrained Weight Vector:", model.w)
    print("Trained Bias Term:", model.b)

    print("\nInference Output:")
    for sample in X_data:
        pred = model.forward(sample)
        print(f"Input Vector: {sample} -> Model Prediction: {pred}")