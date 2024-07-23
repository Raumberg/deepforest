import jax
import jax.numpy as jnp


def xentropy(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(labels * log_probs, axis=-1)