import torch
import torch.nn as nn
import pytorch_lightning as pl
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

        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target_batch.view(-1)

        ignore_index = self.criterion.ignore_index
        valid_mask = target_flat != ignore_index
        num_valid_tokens = valid_mask.sum().item()

        if num_valid_tokens == 0:
            # no valida tokens ->  skip loss for this batch
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        loss = self.criterion(logits_flat, target_flat)

        # handle NaN / Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Warning] NaN/Inf loss at step {self.global_step}, setting to 0.0")
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)


        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_valid_tokens", num_valid_tokens, prog_bar=False, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target_batch.view(-1)
        
        ignore_index = self.criterion.ignore_index
        valid_mask = target_flat != ignore_index
        num_valid_tokens = valid_mask.sum().item()
        
        if num_valid_tokens == 0:
            val_loss = torch.tensor(0.0, device=logits.device)
            self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return val_loss
        
        loss = self.criterion(logits_flat, target_flat)
        
        if torch.isnan(loss):
            print("NaN loss encountered in validation step.")
            loss = torch.tensor(0.0, device=logits.device)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_valid_tokens", num_valid_tokens, prog_bar=False, logger=True)
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
