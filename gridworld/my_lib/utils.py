from pathlib import Path
import yaml
from scipy.linalg import norm as linorm
import numpy as np


EPS = 1e-24

def read_config(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open('r') as config_io:
        return yaml.load(config_io, yaml.Loader)

def normalize(x, default_values=None, return_zeroed_variables_count=False, order=1, skip_zeros=False):
    if len(x.shape) == 1:
        x = x[None]

    norm_x = x.copy()

    if order == 1:
        norm = x.sum(axis=-1)
    else:
        norm = linorm(x, ord=order, axis=-1)

    mask = norm == 0

    if skip_zeros:
        norm[mask] = 1
    else:
        if default_values is None:
            default_values = np.ones_like(x)

        norm_x[mask] = default_values[mask]
        norm[mask] = norm_x[mask].sum(axis=-1)

    if return_zeroed_variables_count:
        return norm_x / norm.reshape((-1, 1)), np.sum(mask)
    else:
        return norm_x / norm.reshape((-1, 1))
