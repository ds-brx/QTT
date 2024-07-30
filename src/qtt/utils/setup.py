import logging
import os
import random
from datetime import datetime
from pathlib import Path
from time import sleep

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_outputdir(path, create_dir=True, warn_if_exist=True, path_suffix=None):
    if path:
        assert isinstance(
            path, (str, Path)
        ), f"Only str and pathlib.Path types are supported for path, got {path} of type {type(path)}."

    if path_suffix is None:
        path_suffix = ""
    if path is None:
        path = f"qtt{path_suffix}"
    else:
        path = f"{path}{path_suffix}"

    if create_dir:
        for _ in range(1000):
            try:
                now = datetime.now()
                timestamp = now.strftime("%y%m%d-%H%M%S")
                _path = os.path.join(path, f"{timestamp}")
                os.makedirs(_path, exist_ok=False)
                path = _path
                break
            except FileExistsError:
                sleep(1)
                continue
        else:
            raise RuntimeError("Too many jobs started at the same time.")
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
    elif warn_if_exist:
        if os.path.isdir(path):
            logger.warning(
                f'Warning: path already exists! This may overwrite previous runs! path="{path}"'
            )
    path = os.path.expanduser(path)  # replace ~ with absolute path if it exists
    return path


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
