# trainer.py
# Full training script for TwinGPT using PyTorch Lightning and your ChatDataModule
# Author: Intistark(modified from Raschka LLM tutorial)
# ------------------------------------------------------------

import json
import pytorch_lightning as pl
import torch
import tiktoken

from twin_gpt.dataset.instruct_datamodule import InstructDataModule
from twin_gpt.models.lit_gpt import LitGPT

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="twin_gpt_instruct_project",
    name="exp_1",
    log_model="all",
)

with open("configs/instruct_gpt2.json", "r") as f:
    cfg = json.load(f)


def main():
    data_path = cfg["data"]["data_path"]
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # data = data[:1000]  # Use a subset for quick testing
    train_portion = int(0.90 * len(data))
    train_data = data[0:train_portion]
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
        save_top_k=2,
        mode="min"
    )


    data_module = InstructDataModule(
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
    
    trainer.save_checkpoint("logs/checkpoints/twin_gpt_final.ckpt")
    torch.save(model.state_dict(), "logs/checkpoints/twin_gpt_final.pth")
    print("Training completed, model saved as twin_gpt_final.pth")

if __name__ == "__main__":
    main()
