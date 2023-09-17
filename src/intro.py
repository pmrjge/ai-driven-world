import os
import math
import numpy as np
import time

import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")

from matplotlib.colors import to_rgba

import seaborn as sns

sns.set()

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import jax.random as jrnd

print("Using jax", jax.__version__)

a = jnp.zeros((2, 5), dtype=jnp.float32)
print(a)

b = jnp.arange(6)
print(b)

print(b.device())

print(jax.devices())

b_new = b.at[0].set(1)

# random numbers
rng = jrnd.PRNGKey(42)

jax_rn1 = jrnd.normal(rng)
jax_rn2 = jrnd.normal(rng)

print("JAX - Random number 1:", jax_rn1)
print("JAX - Random number 2:", jax_rn2)

np.random.seed(42)
np_rn1 = np.random.normal()
np_rn2 = np.random.normal()
print("Numpy - Random number 1:", np_rn1)
print("Numpy - Random number 2:", np_rn2)

rng, sk1, sk2 = jrnd.split(rng, num=3)
jax_rn1 = jrnd.normal(sk1)
jax_rn2 = jrnd.normal(sk2)

print("JAX - Random number 1:", jax_rn1)
print("JAX - Random number 2:", jax_rn2)


