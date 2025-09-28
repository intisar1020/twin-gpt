# train.py
# Full training script for TwinGPT using PyTorch Lightning and your ChatDataModule
# Author: Intistark(modified from Raschka LLM tutorial)
# ------------------------------------------------------------


import os
import pytorch_lightning as pl
import torch
import tiktoken
from functools import partial

from twin_gpt.data.text_datamodule import ChatDataModule
from twin_gpt.models.lit_gpt import LitGPT

cfg = {
    "model": {
        "vocab_size": 50257,  # GPT-2 vocab size
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "train": {
        "lr": 3e-4,
        "epoch": 1,  
        "batch_size": 2,  # gpu poor.
        "weight_decay": 1e-2,
    },
    "data": {
        "train_data_path": "data/train.txt",  # Path to your training data
        "val_data_path": "data/val.txt",      # Path to your validation data
        "batch_size": 2,                      # Reduced for memory constraints; increase as needed
        "num_workers": 4,
        "pad_token_id": 50256,                # GPT-2 pad token ID
        "ignore_index": -100,                 # For ignoring padding in loss computation
        "max_length": 1024,                   # Max sequence length
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "trainer": {
        "max_epochs": 3,                      # Reduced for quicker testing; increase as needed
        "gpus": 1 if torch.cuda.is_available() else 0,
        "precision": 16 if torch.cuda.is_available() else 32,  # Use mixed precision if on GPU
        "log_every_n_steps": 10,
    }
}

model = LitGPT(cfg["model"], cfg["train"])
print(model)