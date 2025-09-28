
from functools import partial
from importlib.metadata import version
import json
import os
import re
import time
import urllib

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn as nn

class GPTLightningModule(pl.LightningDataModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        weight_decay: float = 0.05,
        num_epochs: int = 10,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log('train_loss', loss)
        return loss
        
    