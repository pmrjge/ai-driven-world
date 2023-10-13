import os
import json
import math
import numpy as np
from typing import Any, Sequence
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import optax

import urllib.request
from urllib.error import HTTPError

DATASET_PATH = "../../data"
CHECKPOINT_PATH = "../../saved_models/tutorial3_jax"


base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial3/"

pretrained_files = ["FashionMNIST_elu.config", "FashionMNIST_elu.tar",
                    "FashionMNIST_leakyrelu.config", "FashionMNIST_leakyrelu.tar",
                    "FashionMNIST_relu.config", "FashionMNIST_relu.tar",
                    "FashionMNIST_sigmoid.config", "FashionMNIST_sigmoid.tar",
                    "FashionMNIST_swish.config", "FashionMNIST_swish.tar",
                    "FashionMNIST_tanh.config", "FashionMNIST_tanh.tar"]


# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please contact the author with the full output including the following error:\n", e)


# Common activation functions
class Sigmoid(nn.Module):
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))


class Tanh(nn.Module):
    def __call__(self, x):
        x_exp, neg_x_exp = jnp.exp(x), jnp.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)


class ReLU(nn.Module):
    def __call__(self, x):
        return jnp.maximum(0, x)


class LeakyReLU(nn.Module):
    alpha: float = 0.1

    def __call__(self, x):
        return jnp.where(x>0, x, self.alpha * x)


class ELU(nn.Module):
    def __call__(self, x):
        return jnp.where(x>0, x, jnp.exp(x) - 1)


class Swish(nn.Module):
    def __call__(self, x):
        return x * nn.sigmoid(x)


act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish,
}


# Visualizing activation functions

def get_grads(act_fn, x):
    return jax.vmap(jax.grad(act_fn))(x)


def vis_act_fn(act_fn, ax, x):
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)
    ax.plot(x, y, linewidth=2, label="ActFn")
    ax.plot(x, y_grads, linewidth=2, label="Gradient")
    ax.set_title(act_fn.__class__.__name__)
    ax.legend()
    ax.set_ylim(-1.5, x.max())


act_fns = [act_fn() for act_fn in act_fn_by_name.values()]
x = np.linspace(-5, 5, 1000)
rows = math.ceil(len(act_fns) / 2.0)
fig, ax = plt.subplots(rows, 2, figsize=(8, rows*4))

for i, act_fn in enumerate(act_fns):
    vis_act_fn(act_fn, ax[divmod(i, 2)], x)
fig.subplots_adjust(hspace=0.3)
plt.show()


# Analysing the effect of activation functions
init_fn = lambda x: (lambda rng, shape, dtype: random.uniform(rng, shape=shape, minval=-1/np.sqrt(x.shape[1]), max_val=1/np.sqrt(x.shape[1]), dtype=dtype))


# Network
class BaseNetwork(nn.Module):
    act_fn: nn.Module
    num_classes: int = 10
    hidden_sizes: Sequence = (512, 256, 256, 128)

    @nn.compact
    def __call__(self, x, return_activations=False):
        x = x.reshape(x.shape[0], -1)
        activations = []
        for hd in self.hidden_sizes:
            x = nn.Dense(hd, kernel_init=init_fn(x), bias_init=init_fn(x))(x)
            activations.append(x)

        x = nn.Dense(self.num_classes, kernel_init=init_fn(x), bias_init=init_fn(x))(x)
        return x if not return_activations else (x, activations)


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")

def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")

def load_model(model_path, model_name, state=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        state - (Optional) If given, the parameters are loaded into this training state. Otherwise,
                a new one is created alongside a network architecture.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if state is None:
        act_fn_name = config_dict["act_fn"].pop("name").lower()
        act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
        net = BaseNetwork(act_fn=act_fn, **config_dict)
        state = train_state.TrainState(step=0,
                                       params=None,
                                       apply_fn=net.apply,
                                       tx=None,
                                       opt_state=None)
    else:
        net = None
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead load the parameters simply from a pickle file.
    with open(model_file, 'rb') as f:
        params = pickle.load(f)
    state = state.replace(params=params)
    return state, net

def save_model(model, params, model_path, model_name):
    """
    Given a model, we save the parameters and hyperparameters.

    Inputs:
        model - Network object without parameters
        params - Parameters to save of the model
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = {
        'num_classes': model.num_classes,
        'hidden_sizes': model.hidden_sizes,
        'act_fn': {'name': model.act_fn.__class__.__name__.lower()}
    }
    if hasattr(model.act_fn, 'alpha'):
        config_dict['act_fn']['alpha'] = model.act_fn.alpha
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead save the parameters simply in a pickle file.
    with open(model_file, 'wb') as f:
        pickle.dump(params, f)


import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - 0.5) / 0.5
    return img


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


train_dataset = FashionMNIST(root=DATASET_PATH, train=True, transform=image_to_numpy, download=True)

train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))

test_set = FashionMNIST(root=DATASET_PATH, train=False, transform=image_to_numpy, download=True)

train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate)

val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate)
test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate)

# visualizing the gradient flow after initialization

