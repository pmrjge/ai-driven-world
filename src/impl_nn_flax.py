import flax
from flax import linen as nn
import jax


class SimpleClassifier(nn.Module):
    num_hidden: int
    num_outputs: int

    def setup(self):
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x


class SimpleClassifierCompact(nn.Module):
    num_hidden: int
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x


# training a NN

model = SimpleClassifierCompact(num_hidden=8, num_outputs=1)

rng = jax.random.PRNGKey(42)
rng, inp_rng, init_rng = jax.random.split(rng, 3)

inp = jax.random.normal(inp_rng, (8, 2))
params = model.init(init_rng, inp)
print(params)
