import torch
from pathlib import Path
from config import Config
from utility.load_save import load

def predict(cfg:Config):
    settings = load()