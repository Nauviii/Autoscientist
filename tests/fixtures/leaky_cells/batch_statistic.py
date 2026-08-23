class FeatureStep:
    """Leaky: percentile rank recomputed from whatever batch transform receives."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        out = X.copy()
        out["num_0_rank"] = X["num_0"].rank(pct=True)
        return out