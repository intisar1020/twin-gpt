# credit: most of the source and function are adopted from https://github.com/rasbt/LLMs-from-scratch
# please refer to https://github.com/rasbt/LLMs-from-scratch for original source.


import re
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
from typing import Optional, Callable, List




ASSISTANT_NAME = "intisar chowdhury"  # canonical assistant name (case-insensitive)
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"


class ChatDataset(Dataset):
    """
    Produces (input_ids, labels) pairs.
    Labels are -100 for tokens that should be ignored (user/context + padding).
    """
    def __init__(self, raw_lines: list[str], tokenizer, max_length: int = None):
        """
        raw_lines: list of lines from your conf.txt (non-empty, stripped)
        tokenizer: tiktoken encoding
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_text = []
        file_for_debug = open("debug-dataset.txt", "w", encoding="utf-8")
        
        for i in range(0, len(raw_lines) - 1, 2):
            raw_a = raw_lines[i]
            raw_b = raw_lines[i + 1]

            speaker_a, msg_a = self._split_speaker_and_text(raw_a)
            speaker_b, msg_b = self._split_speaker_and_text(raw_b)
            if ("called" in msg_a) or ("called" in msg_b):
                continue
            if("missed your call." in msg_a) or ("missed your call." in msg_b):
                continue
            
            if (speaker_a is None) or (speaker_b is None):
                continue
            # if (not "Intisar Chowdhury" in speaker_a) or not ("Intisar Chowdhury" in speaker_b):
            #     continue

            # Decide which line is user and which is assistant.
            # If one speaker matches ASSISTANT_NAME -> that's assistant.
            # If both are users (no assistant), treat a as user, b as assistant (best-effort).
            # If both assistant, skip (no user).
            is_a_assistant = (speaker_a is not None and speaker_a.lower() == ASSISTANT_NAME)
            is_b_assistant = (speaker_b is not None and speaker_b.lower() == ASSISTANT_NAME)

            if is_a_assistant and not is_b_assistant:
                assistant_text = msg_a
                user_text = msg_b
            elif is_b_assistant and not is_a_assistant:
                assistant_text = msg_b
                user_text = msg_a
            elif not is_a_assistant and not is_b_assistant:
                # no explicit assistant found -> treat line A as user, line B as assistant (best-effort)
                user_text = msg_a
                assistant_text = msg_b
            else:
                # both assistant: to not waste dataset we treat line A as user, line B as assistant
                user_text = msg_a
                assistant_text = msg_b

            # Clean texts
            user_text = self.clean_text(user_text)
            assistant_text = self.clean_text(assistant_text)

            # Format in LLaMA-style
            prefix = f"{USER_TAG}\n{user_text}\n{ASSISTANT_TAG}\n" # this will be provided as context.
            full_text = prefix + assistant_text  # no trailing newline necessary

            full_ids = tokenizer.encode(full_text)
            self.encoded_text.append(full_ids)
            # save a copy with text for debugging in a file.
            file_for_debug.write(full_text + "\n")

        file_for_debug.close()

    def _split_speaker_and_text(self, line: str):
        """
        Attempt to split "Name: message". If no colon, return (None, line).
        """
        if ":" in line:
            name, msg = line.split(":", 1)
            return name.strip(), msg.strip()
        else:
            return None, line.strip()

    def clean_text(self, s: str) -> str:
        # Keep only basic printable characters
        s = re.sub(r"[^\x20-\x7E]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def __len__(self):
        return len(self.encoded_text)

    def __getitem__(self, idx):
        return self.encoded_text[idx]
    

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst = []
    labels_lst = []

    for item in batch:
        new_item = item.copy() + [pad_token_id]
        new_item = new_item[:batch_max_length]
        pad_len = batch_max_length - len(new_item)
        new_item = new_item + [pad_token_id] * pad_len

        inputs = torch.tensor(new_item[:-1], dtype=torch.long)
        labels = torch.tensor(new_item[1:], dtype=torch.long)

        pad_indices = (labels == pad_token_id).nonzero(as_tuple=True)[0]
        if pad_indices.numel() > 1:
            labels[pad_indices[1:]] = ignore_index

        inputs_lst.append(inputs)
        labels_lst.append(labels)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    labels_tensor = torch.stack(labels_lst).to(device)
    
    return inputs_tensor, labels_tensor


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