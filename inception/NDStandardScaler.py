# from https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import numpy as np

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, x, **kwargs):
        x = np.array(x)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(x.shape) > 1:
            self._orig_shape = x.shape[1:]
        x = self._flatten(x)
        self._scaler.fit(x, **kwargs)
        return self

    def transform(self, x, **kwargs):
        x = np.array(x)
        x = self._flatten(x)
        x = self._scaler.transform(x, **kwargs)
        x = self._reshape(x)
        return x

    def _flatten(self, x):
        # Reshape X to <= 2 dimensions
        if len(x.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            x = x.reshape(-1, n_dims)
        return x

    def _reshape(self, x):
        # Reshape X back to it's original shape
        if len(x.shape) >= 2:
            x = x.reshape(-1, *self._orig_shape)
        return x