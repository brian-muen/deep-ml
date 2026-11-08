import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""

	m, n = X.shape
	X = np.column_stack((np.ones(m), X))
	betas = np.zeros(n + 1)
	losses = []

	for _ in range(iterations):
		predictions = 1 / (1 + np.exp(-(X @ betas)))
		errors = predictions - y

		loss = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
		losses.append(round(float(loss), 4))

		betas = betas - learning_rate * (X.T @ errors) 


	return (np.round(betas, 4).tolist(), losses)
	