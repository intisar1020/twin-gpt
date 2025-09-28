# trainer.py
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


import wandb
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="twin_gpt_project",
    name="twin_gpt_training",
    log_model="all",
)

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
        "epoch": 3,  
        "batch_size": 4,  # GPU memory poor
        "weight_decay": 1e-2,
        "pretrained_path": "./ckpts/gpt2-124M.pth", 
        "warmup_steps": 500,  # linear warmup
    },
    "data": {
        "data_path": "data/conf.txt",  
        "batch_size": 4,                      
        "num_workers": 0,
        "pad_token_id": 50256,                
        "ignore_index": -100,                 
        "max_length": 1024,                   
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "trainer": {
        "max_epochs": 3,                
        "log_every_n_steps": 5,
        "devices": 1 if torch.cuda.is_available() else None,
        "precision": 16,                # mixed precision
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
    }
}


def main():
    data_path = cfg["data"]["data_path"]
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    train_portion = int(0.9 * len(data))
    train_data = data[:train_portion]
    val_data = data[train_portion:]
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    tokenizer = tiktoken.get_encoding("gpt2")
    try:
        pad_token_id = tokenizer.encode("<|endoftext|>")[0]
    except Exception:
        pad_token_id = 50256  # default GPT-2 pad token ID


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="logs/checkpoints",
        filename="twin_gpt-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )

   
    data_module = ChatDataModule(
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pad_token_id=pad_token_id,
        ignore_index=cfg["data"]["ignore_index"],
        max_length=cfg["data"]["max_length"],
        device=cfg["data"]["device"],
    )

    # Model
    model = LitGPT(cfg["model"], cfg["train"])

    # Trainer
    trainer = pl.Trainer(
        **cfg["trainer"],
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    # Train
    trainer.fit(model, data_module)


    torch.save(model.state_dict(), "twin_gpt_final.pth")
    print("Training completed, model saved as twin_gpt_final.pth")

if __name__ == "__main__":
    main()
