import flax
import numpy as np
from flax import linen as nn
import jax
import torch.utils.data as data
import matplotlib.pyplot as plt
import optax
from flax.training import train_state, checkpoints
from jax import numpy as jnp
from tqdm import tqdm

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

# Define Dataset for training


class XORDataset(data.Dataset):
    def __init__(self, size, seed, std=0.1):
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(
            np.float32
        )
        label = (data.sum(axis=1) == 1).astype(np.int32)

        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


dataset = XORDataset(size=200, seed=4)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])


def visualize_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()


visualize_samples(dataset.data, dataset.label)
plt.show()


# For the dataloader class
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate)

data_inputs, data_labels = next(iter(data_loader))

print("Data inputs", data_inputs.shape, "\n", data_inputs)
print("Data labels", data_labels.shape, "\n", data_labels)


# Optimization
optimizer = optax.sgd(learning_rate=0.1)

model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# Loss Function
def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)

    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc

batch = next(iter(data_loader))
print(calculate_loss_acc(model_state, model_state.params, batch))

# Creating an efficient training and validation step

@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss_acc, argnums=1, has_aux=True)

    (loss,acc), grads = grad_fn(state, state.params, batch)

    state = state.apply_gradients(grads=grads)

    return state, loss, acc

@jax.jit
def eval_step(state, batch):
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

# Training
train_dataset = XORDataset(size=2500, seed=42)
train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)

def train_model(state, data_loader, num_epochs=100):
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
    return state

trained_model_state = train_model(model_state, train_dataloader, num_epochs=100)

ckpt_dir='my_checkpoints/'
prefix='my_model'
checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=trained_model_state, step=100, prefix=prefix, overwrite=True)

loaded_model_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=model_state, prefix=prefix)

# Evaluation