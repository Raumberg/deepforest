import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from optax import Adam


class Tree(nn.Module):
    depth: int
    n_in_feature: int
    used_feature_rate: float
    n_class: int
    jointly_training: bool

    @nn.compact
    def __call__(self, x):
        feature_mask = jnp.eye(self.n_in_feature)[:, :int(self.n_in_feature * self.used_feature_rate)]
        x = jnp.dot(x, feature_mask)
        decision = nn.Dense(features=2 ** self.depth)(x)
        decision = nn.sigmoid(decision)
        decision = jnp.concatenate([decision, 1 - decision], axis=-1)
        mu = jnp.ones((x.shape[0], 1, 1))
        for _ in range(self.depth):
            mu = mu * decision[:, 2 ** _ : 2 ** (_ + 1), :]
        mu = mu.reshape(x.shape[0], -1)
        return mu

    def get_pi(self):
        if self.jointly_training:
            return jax.nn.softmax(self.pi, axis=-1)
        else:
            return self.pi

    def cal_prob(self, mu, pi):
        return jnp.dot(mu, pi)

    def update_pi(self, new_pi):
        self.pi = new_pi

class Forest(nn.Module):
    n_tree: int
    tree_depth: int
    n_in_feature: int
    tree_feature_rate: float
    n_class: int
    jointly_training: bool

    @nn.compact
    def __call__(self, x):
        trees = []
        for _ in range(self.n_tree):
            tree = Tree(self.tree_depth, self.n_in_feature, self.tree_feature_rate, self.n_class, self.jointly_training)
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            trees.append(p[:, :, jnp.newaxis])
        trees = jnp.concatenate(trees, axis=2)
        prob = jnp.sum(trees, axis=2) / self.n_tree
        return prob

class NeuralDecisionForest(nn.Module):
    feature_layer: nn.Module
    forest: nn.Module

    def __init__(self, feature_layer, forest):
        self.feature_layer = feature_layer
        self.forest = forest
        self.optim = Adam(learning_rate=0.001)

    @nn.compact
    def __call__(self, x):
        x = self.feature_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.forest(x)
        return x

    @property
    def get_optim(self):
        return self.optim

    def set_optim(self, optimizer):
        self.optim = optimizer

    @staticmethod
    def loss_fn(model, params, x, y):
        logits = model.apply(params, x)
        loss = jnp.mean(jnp.sum(jax.nn.softmax_cross_entropy_with_logits(logits, y), axis=-1))
        return loss

    @staticmethod
    def train(self, params, x, y, epochs):
        opt_state = self.optim.init(params)
        for i in range(epochs):
            loss, grads = jax.value_and_grad(self.loss_fn)(params, x, y)
            updates, opt_state = self.optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            print(f"Epoch {i+1}, Loss: {loss}")
        return params