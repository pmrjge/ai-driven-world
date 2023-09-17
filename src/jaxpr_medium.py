import jax
import jax.numpy as jnp
import jax.random as jr

from timeit import default_timer as time
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")
from matplotlib.colors import to_rgba

import seaborn as sns

sns.set()


def simple_graph(x):
    x = x + 2
    x = x**2
    x = x + 3
    return x.mean()


inp = jnp.arange(3, dtype=jnp.float32)
print(jax.make_jaxpr(simple_graph)(inp))

# Automatic Differentiation

grad_fn = jax.grad(simple_graph)
gradients = grad_fn(inp)
print("Gradient", gradients)

print(jax.make_jaxpr(grad_fn)(inp))

val_grad_fn = jax.value_and_grad(simple_graph)
print(val_grad_fn(inp))

# Speeding up with jit
jitted_fn = jax.jit(simple_graph)

key = jr.PRNGKey(42)
rng, normal_rng = jr.split(key)

large_input = jr.normal(normal_rng, (1000,))
_ = jitted_fn(large_input)


def performance(f, x, s="Default"):
    t0 = time()
    f(x).block_until_ready()
    t1 = time()
    print(f"{s}: Time elapsed {t1 - t0} ms")


performance(simple_graph, large_input, "simple graph")
performance(jitted_fn, large_input, "jitted")

# Applying/ Compiling JIT for gradient functions

jitted_grad_fn = jax.jit(grad_fn)
_ = jitted_grad_fn(large_input)

performance(grad_fn, large_input, "grad_fn")
performance(jitted_grad_fn, large_input, "jitted_grad_fn")
