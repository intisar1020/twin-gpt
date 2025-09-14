# credit: most of the source and function are adopted from https://github.com/rasbt/LLMs-from-scratch
# please refer to https://github.com/rasbt/LLMs-from-scratch for original source.


import re
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
from typing import Optional, Callable, List

class ChatDataset(Dataset):
    """Chat dataset for user-assistant message pairs.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data: list[str], tokenizer: callable):
        self.data = data
        self.encoded_texts = []
        
        for i in range(0, len(data)- 1, 2):
            user_message = self.clean_text(data[i])
            assistant_message = self.clean_text(data[i + 1])
            self.encoded_texts.append(
                tokenizer.encode(user_message + "\n" + assistant_message)
            )
    
    def clean_text(self, s: str) -> str:
        # keeping only basic printable characters
        s = re.sub(r"[^\x20-\x7E]+", "", s)
        return s.strip()

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.encoded_texts)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

class ChatDataModule(pl.LightningDataModule):
    """Lightning DataModule for chat datasets."""

    def __init__(
        self,
        tokenizer,
        train_data: list[str],
        val_data: list[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pad_token_id: int = 50256,
        ignore_index: int = -100,
        max_length: int = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.device = device

    def setup(self, stage=None):
        self.train_dataset = ChatDataset(self.train_data, self.tokenizer)
        self.val_dataset = (
            ChatDataset(self.val_data, self.tokenizer) if self.val_data else None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(
                custom_collate_fn,
                pad_token_id=self.pad_token_id,
                ignore_index=self.ignore_index,
                allowed_max_length=self.max_length,
                device=self.device,
            ),
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=partial(
                    custom_collate_fn,
                    pad_token_id=self.pad_token_id,
                    ignore_index=self.ignore_index,
                    allowed_max_length=self.max_length,
                    device=self.device,
                ),
            )
        return None