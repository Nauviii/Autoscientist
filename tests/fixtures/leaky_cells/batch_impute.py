class FeatureStep:
    """Leaky: imputation uses the mean of the batch being transformed."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        out = X.copy()
        out["num_1"] = out["num_1"].fillna(out["num_1"].mean())
        return out