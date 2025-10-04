import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from .gpt_model import GPTModel

class LitGPT(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPTModel(model_cfg)

        self.model.load_state_dict(torch.load(train_cfg.get("pretrained_path", "twin_gpt_initial.pth")))
        
        self.lr = train_cfg["lr"]
        self.epoch = train_cfg["epoch"]
        self.weight_decay = train_cfg["weight_decay"]
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        return self.model(x)

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
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00005, weight_decay=0.1)
        # def lr_lambda(current_step: int):
        #     if current_step < self.warmup_steps:
        #         return float(current_step) / float(max(1, self.warmup_steps))
        #     return max(
        #         0.0, 0.5 * (1.0 + torch.cos(torch.tensor(current_step - self.warmup_steps) / (self.epoch*1000) * 3.1415926535))
        #     )

        # scheduler = {
        #     'scheduler': LambdaLR(optimizer, lr_lambda),
        #     'interval': 'step',
        #     'frequency': 1
        # }
        return [optimizer]# , [scheduler]
