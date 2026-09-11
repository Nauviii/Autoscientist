class FeatureStep:
    """Clean control: every statistic is learned in fit and frozen at transform."""

    def fit(self, X, y):
        self.median_ = X["num_1"].median()
        self.quantiles_ = X["num_0"].quantile([0.25, 0.5, 0.75]).tolist()
        return self

    def transform(self, X):
        out = X.copy()
        out["num_1"] = out["num_1"].fillna(self.median_)
        out["num_0_bucket"] = sum((out["num_0"] > q).astype(int) for q in self.quantiles_)
        return out