import numpy as np


"""
Implementation of Principal Component Analysis.
"""


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        # Calculate the mean of the training data
        self.mean = np.mean(X, axis=0)

        # Subtract the mean from the data
        X_centered = X - self.mean

        # Calculate the covariance matrix
        # Use rowvar=False because each row is a data point
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[: self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet. Call fit() first.")

        # Subtract the mean from the data
        X_centered = X - self.mean

        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components)
        return X_transformed

    def reconstruct(self, X_transformed: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet. Call fit() first.")

        # Project the data back onto the original feature space
        X_reconstructed = np.dot(X_transformed, self.components.T) + self.mean
        return X_reconstructed
