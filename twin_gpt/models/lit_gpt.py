import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .gpt_model import GPTModel

class LitGPT(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPTModel(model_cfg)
        self.lr = train_cfg["lr"]
        self.epoch = train_cfg["epoch"]
        self.weight_decay = train_cfg["weight_decay"]
        self.scheduler = train_cfg.get("scheduler", None)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'train_loss',
                'frequency': 1
            }
        }