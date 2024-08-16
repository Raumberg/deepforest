import jax
import jax.numpy as jnp
from flax import linen as nn


class MNISTFeatureLayer(nn.Module):
    dropout_rate: float
    shallow: bool

    @nn.compact
    def __call__(self, x):
        if self.shallow:
            x = nn.Conv(features=64, kernel_size=(15, 15), padding=(1, 1), strides=(5, 5))(x)
        else:
            x = nn.Conv(features=32, kernel_size=(3, 3), padding=(1, 1))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.dropout(x, rate=self.dropout_rate)
            x = nn.Conv(features=64, kernel_size=(3, 3), padding=(1, 1))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.dropout(x, rate=self.dropout_rate)
            x = nn.Conv(features=128, kernel_size=(3, 3), padding=(1, 1))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.dropout(x, rate=self.dropout_rate)
        return x

    def get_out_feature_size(self):
        if self.shallow:
            return 64 * 4 * 4
        else:
            return 128 * 3 * 3

class UCIAdultFeatureLayer(nn.Module):
    dropout_rate: float
    shallow: bool

    @nn.compact
    def __call__(self, x):
        if self.shallow:
            x = nn.Dense(features=1024)(x)
        else:
            raise NotImplementedError
        return x

    def get_out_feature_size(self):
        return 1024

class UCILetterFeatureLayer(nn.Module):
    dropout_rate: float
    shallow: bool

    @nn.compact
    def __call__(self, x):
        if self.shallow:
            x = nn.Dense(features=1024)(x)
        else:
            raise NotImplementedError
        return x

    def get_out_feature_size(self):
        return 1024

class UCIYeastFeatureLayer(nn.Module):
    dropout_rate: float
    shallow: bool

    @nn.compact
    def __call__(self, x):
        if self.shallow:
            x = nn.Dense(features=1024)(x)
        else:
            raise NotImplementedError
        return x

    def get_out_feature_size(self):
        return 1024