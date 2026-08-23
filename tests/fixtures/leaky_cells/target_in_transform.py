class FeatureStep:
    """Leaky: stores the training target and reuses it at transform time."""

    def fit(self, X, y):
        self.y_ = y
        return self

    def transform(self, X):
        out = X.copy()
        out["target_echo"] = self.y_.reindex(X.index).fillna(self.y_.mean())
        return out
