from sklearn.covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from numbers import Real
import numpy as np
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from time import time
from scipy.linalg import eig
from numpy import ndarray
import math


def power_method(A, num_iter=1000, tol=1e-6):
    """
    Finds the largest eigenvalue and corresponding eigenvector using the Power Method.
    """
    n = A.shape[0]
    b_k = np.random.rand(n)

    for _ in range(num_iter):
        # Multiply by the matrix
        b_k1 = np.dot(A, b_k)

        # Normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        b_k1 = b_k1 / b_k1_norm

        # Check convergence
        if np.linalg.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    # Compute the Rayleigh quotient for the eigenvalue
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))

    return eigenvalue, b_k

def deflation(A, eigenvalue, eigenvector):
    """
    Performs deflation to remove the influence of a given eigenvalue and eigenvector.
    """
    return A - eigenvalue * np.outer(eigenvector, eigenvector)

def find_k_largest_eigenvalues(A, k, num_iter=1000, tol=1e-6):
    """
    Finds the k largest eigenvalues and their corresponding eigenvectors using the
    Power Method with Deflation.
    """
    n = A.shape[0]
    assert k <= n, "k cannot be larger than the size of the matrix"

    eigenvalues = []
    eigenvectors = []

    A_copy = A.copy()

    for _ in range(k):
        # Find the largest eigenvalue and eigenvector
        eigenvalue, eigenvector = power_method(A_copy, num_iter, tol)

        # Store the results
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        # Deflate the matrix
        A_copy = deflation(A_copy, eigenvalue, eigenvector)

    return np.array(eigenvalues), np.array(eigenvectors).T

def size(x):
    return math.prod(x.shape)

def compute_scatter_matrices(X, y):
    """
    Compute Within-Class Scatter (S_W) and Between-Class Scatter (S_B).
    
    Parameters:
    X : ndarray
        Feature matrix (n_samples, n_features).
    y : ndarray
        Class labels (n_samples,).
        
    Returns:
    S_W : ndarray
        Within-class scatter matrix (n_features, n_features).
    S_B : ndarray
        Between-class scatter matrix (n_features, n_features).
    """
    # Step 1: Compute overall mean
    overall_mean = np.mean(X, axis=0)
    
    # Step 2: Initialize scatter matrices
    n_features = X.shape[1]
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))
    
    # Step 3: Compute scatter matrices
    classes = np.unique(y)
    for _class in classes:
        # Extract samples of the current class
        X_class = X[y == _class]
        
        # Compute mean vector for the current class
        class_mean = np.mean(X_class, axis=0)
        
        # Compute within-class scatter (S_W)
        n_class_samples = X_class.shape[0]
        X_class_centered = X_class - class_mean  # Center the data
        S_W += X_class_centered.T @ X_class_centered
        
        # Compute between-class scatter (S_B)
        mean_diff = (class_mean - overall_mean).reshape(-1, 1)
        S_B += n_class_samples * (mean_diff @ mean_diff.T)
    
    return S_W, S_B


class LDA:

    def __init__(
        self,

        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        covariance_estimator=None,
    ):
        self.solver = 'eigen'
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver
        self.covariance_estimator = covariance_estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, _ = X.shape
        n_classes = self.classes_.shape[0]

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        if self.priors is None:  # estimate priors from sample
            _, cnts = np.unique(y, return_counts=True)  # non-negative ints
            self.priors_ = (cnts / float(y.shape[0])).astype(X.dtype)
        else:
            self.priors_ = np.asarray(self.priors, dtype=X.dtype)

        if np.any(self.priors_ < 0):
            raise ValueError("priors must be non-negative")

        if np.abs(np.sum(self.priors_) - 1.0) > 1e-5:
            self.priors_ = self.priors_ / self.priors_.sum()

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(n_classes - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        self._solve_eigen(
            X,
            y,
            shrinkage=self.shrinkage,
            covariance_estimator=self.covariance_estimator,
        )

        self._n_features_out = self._max_components
        return self

    def _solve_eigen(self, X, y, shrinkage=None, covariance_estimator=None):
        Sw, Sb = compute_scatter_matrices(X, y)
        evals, evecs = find_k_largest_eigenvalues(np.linalg.inv(Sw).dot(Sb), self._max_components)
        self.scalings_ = evecs

    def transform(self, X):

        if self.solver == "svd":
            X_new = (X - self.xbar_) @ self.scalings_
        elif self.solver == "eigen":
            X_new = X @ self.scalings_[:, : self._max_components]
        return X_new