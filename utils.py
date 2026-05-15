import os
import yaml
import logging
import numpy as np

# ── Logger ──────────────────────────────────────────────────
_logger = None

def init_logger(args):
    global _logger
    log_dir = args.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)

    log_level = getattr(logging, args.get('state', 'INFO').upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    _logger = logging.getLogger('CET-MAE')

def getLogger():
    global _logger
    if _logger is None:
        logging.basicConfig(level=logging.INFO)
        _logger = logging.getLogger('CET-MAE')
    return _logger

# ── Config Reader ────────────────────────────────────────────
def read_configuration(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# ── Early Stopping ───────────────────────────────────────────
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.min_loss - self.min_delta:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False