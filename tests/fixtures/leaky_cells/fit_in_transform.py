class FeatureStep:
    """Leaky: encoder refitted on the batch instead of on the training fold."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        from sklearn.preprocessing import LabelEncoder

        out = X.copy()
        out["cat_0_code"] = LabelEncoder().fit_transform(X["cat_0"])
        return out