import jax
from jax import numpy as jnp


# Automatic Vectorization with vmap
def simple_linear(x, w, b):
    return (x[:, None] * w).sum(axis=0) + b


rng = jax.random.PRNGKey(42)
rng, x_rng, w_rng, b_rng = jax.random.split(rng, 4)
x_in = jax.random.normal(x_rng, (4,))
w_in = jax.random.normal(w_rng, (4, 3))
b_in = jax.random.normal(b_rng, (3,))

print(simple_linear(x_in, w_in, b_in))


vectorizered_linear = jax.vmap(simple_linear, in_axes=(0, None, None), out_axes=0)

x_vec_in = jnp.stack([x_in] * 5, axis=0)

print(vectorizered_linear(x_vec_in, w_in, b_in))

# Parallel Evaluation with pmap

# jax.pmap

# Working with PyTrees

# parameters = jax.tree_leaves(model_state.params)
# print('We have parameters with the following shapes:', ', '.join([str(p.shape) for p in parameters]))
# print('Overall parameter count:', sum([np.prod(p.shape) for p in parameters])

# jax.tree_map(lambda p: p.shape, model_state.params)

# The Sharp Bits

# Dynamic Shapes


def my_function(x):
    print('Running the function with shape', x.shape)
    return x.mean()


jitted_my_function = jax.jit(my_function)

for i in range(10):
    jitted_my_function(jnp.zeros(i+1,))

for i in range(10):
    jitted_my_function(jnp.zeros(i+1,))

