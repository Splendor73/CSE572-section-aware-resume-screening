import pytest
import numpy as np
from src.clustering.cluster import reduce_dimensions, run_kmeans

def test_reduce_dimensions():
    # Create random matrix
    np.random.seed(42)
    X = np.random.rand(100, 200)
    
    # Reduce dimensions
    X_red = reduce_dimensions(X, n_components=5)
    
    # Check shape
    assert X_red.shape == (100, 5)

def test_run_kmeans():
    # Create fake data that forms distinct clusters
    np.random.seed(42)
    X_class1 = np.random.normal(loc=0.0, scale=0.5, size=(50, 10))
    X_class2 = np.random.normal(loc=10.0, scale=0.5, size=(50, 10))
    X = np.vstack([X_class1, X_class2])
    y_true = np.array([0]*50 + [1]*50)
    
    # Run kmeans with k=2
    results = run_kmeans(X, y_true, k_range=[2])
    
    assert len(results) == 1
    assert results[0]["k"] == 2
    assert results[0]["silhouette"] > 0.5  # Distinct clusters should have high silhouette
    assert results[0]["nmi"] > 0.8  # Should match the ground truth categories well
