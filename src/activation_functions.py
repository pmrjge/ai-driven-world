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

