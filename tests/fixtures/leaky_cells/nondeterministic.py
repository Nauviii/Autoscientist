class FeatureStep:
    """Invalid: unseeded randomness makes the pipeline irreproducible."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        import numpy as np

        out = X.copy()
        out["noise"] = np.random.rand(len(X))
        return out
