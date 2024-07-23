import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax import linen as nn

from utils import xentropy

class NeuralForestLayer(nn.Module):
    """Neural Forest Layer"""
    depth: int
    n_estimators: int
    n_outputs: int
    pi_iters: int

    def setup(self):
        """Initialize pi parameters"""
        self.pi = self.param('pi', jax.nn.initializers.ones, (self.n_estimators * ((1 << self.depth) - 1), self.n_outputs))

    def normalize_pi(self) -> None:
        """Normalize pi parameters"""
        sum_pi_over_y = jnp.sum(self.pi, axis=1, keepdims=True)
        all_0_y = jnp.equal(sum_pi_over_y, 0)
        norm_pi_body = (self.pi + all_0_y * (1.0 / self.n_outputs)) / (sum_pi_over_y + all_0_y)
        self.pi = norm_pi_body

    def update_pi_one_iter(self, leaf_proba: jnp.ndarray, y: jnp.ndarray) -> None:
        """Update pi parameters for one iteration"""
        proba_shaped = leaf_proba.reshape((-1, self.n_estimators * ((1 << self.depth) - 1)))
        y_shaped = y[:, None, None, :]
        common = self.pi[None, :, :] * proba_shaped[:, :, None]
        nominator = common * y_shaped
        denominator = common.sum(axis=1, keepdims=True)
        denominator = denominator + jnp.equal(denominator, 0)
        result = (nominator / denominator).sum(0).reshape((self.n_estimators * ((1 << self.depth) - 1), self.n_outputs))
        self.pi = result

    def update_pi(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """Update pi parameters for multiple iterations"""
        leaf_proba = self.get_probabilities_for(X)
        for _ in range(self.pi_iters):
            self.update_pi_one_iter(leaf_proba, y)
            self.normalize_pi()

    def get_probabilities_for(self, input: jnp.ndarray) -> jnp.ndarray:
        """Compute probabilities for input"""
        lastOffset = 0
        nextOffset = self.n_estimators
        lastTensor = input[:, 0:self.n_estimators]

        for i in range(self.depth - 1):
            lastWidth = (1 << i) * self.n_estimators
            lastOffset, midOffset, nextOffset = nextOffset, nextOffset + lastWidth, nextOffset + lastWidth * 2

            leftTensor = input[:, lastOffset:midOffset]
            rightTensor = input[:, midOffset:nextOffset]

            leftProduct = lastTensor * leftTensor
            rightProduct = (1 - lastTensor) * rightTensor

            lastTensor = jnp.concatenate([leftProduct, rightProduct], axis=1)
        return lastTensor

    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        """Compute output for input"""
        p = self.get_probabilities_for(input)
        return jnp.matmul(p, self.pi)


class ShallowNeuralForest(nn.Module):
    """Shallow Neural Forest Model"""
    n_inputs: int
    n_outputs: int
    regression: bool
    multiclass: bool
    depth: int
    n_estimators: int
    n_hidden: int
    learning_rate: float
    num_epochs: int
    pi_iters: int
    sgd_iters: int
    batch_size: int
    momentum: float
    dropout: float

    def setup(self):
        """Initialize model components"""
        self.l_input = nn.Dense(self.n_hidden, kernel_init=jax.nn.initializers.zeros)
        self.l_forest = NeuralForestLayer(self.depth, self.n_estimators, self.n_outputs, self.pi_iters)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray, X_val: jnp.ndarray = None, y_val: jnp.ndarray = None, on_epoch=None, verbose=False) -> None:
        """Train the model"""
        X = X.astype(jnp.float32)
        y = y.astype(jnp.float32)
        self._x_mean = jnp.mean(X, axis=0)
        self._x_std = jnp.std(X, axis=0)
        self._x_std = jnp.where(self._x_std == 0, 1, self._x_std)
        X = (X - self._x_mean) / self._x_std

        if X_val is not None:
            X_val = X_val.astype(jnp.float32)
            y_val = y_val.astype(jnp.float32)
            X_val = (X_val - self._x_mean) / self._x_std

        loss_fn = xentropy if not self.regression else jax.nn.l2_loss
        optimizer = optax.adam(self.learning_rate)

        @jax.jit
        def update(params, X, y):
            loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
            optimizer.update(grads, params)
            return loss

        for epoch in range(self.num_epochs):
            if epoch % self.sgd_iters == 0:
                self.params['l_forest'].update_pi(X, y)
            loss = 0

            for Xb, yb in jax.utils.batching.batch(X, y, self.batch_size):
                loss += update(self.init_with_output(self.l_forest, Xb, yb), Xb, yb)
            loss /= X.shape[0]

            if X_val is not None:
                predictions = self.l_forest(X_val)
                accuracy = jnp.mean(jnp.equal(jnp.argmax(predictions, axis=1), jnp.argmax(y_val, axis=1)))

                if on_epoch is not None:
                    if X_val is None:
                        on_epoch(epoch, loss)
                    else:
                        on_epoch(epoch, loss, loss, accuracy)
        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions for input"""
        X = X.astype(jnp.float32)
        X = (X - self._x_mean) / self._x_std
        return jnp.argmax(self.l_forest(X), axis=1)

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """Compute probabilities for input"""
        X = X.astype(jnp.float32)
        X = (X - self._x_mean) / self._x_std
        return self.l_forest(X)

    def preprocess_data(self, X: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Preprocess data"""
        self._x_mean = jnp.mean(X, axis=0)
        self._x_std = jnp.std(X, axis=0)
        self._x_std = jnp.where(self._x_std == 0, 1, self._x_std)
        X = (X - self._x_mean) / self._x_std
        if X_val is not None:
            X_val = X_val.astype(jnp.float32)
            y_val = y_val.astype(jnp.float32)
            X_val = (X_val - self._x_mean) / self._x_std
        return X, y