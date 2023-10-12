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
    "swish": Swish
}


# Visualizing activation functions

