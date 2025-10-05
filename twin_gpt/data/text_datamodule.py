# text_datamodule.py
# credit: based on https://github.com/rasbt/LLMs-from-scratch
# Modified to train only on assistant tokens

import re
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
from typing import List, Optional, Tuple


ASSISTANT_NAME = "intisar chowdhury"  # canonical assistant name (case-insensitive)
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"


class ChatDataset(Dataset):
    """
    Produces (input_ids, labels) pairs.
    Labels are -100 for tokens that should be ignored (user/context + padding).
    """
    def __init__(self, raw_lines: List[str], tokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_text: List[Tuple[List[int], List[int]]] = []

        # with open("debug-dataset.txt", "w", encoding="utf-8") as file_for_debug:
        for i in range(0, len(raw_lines) - 1, 2):
            raw_a = raw_lines[i]
            raw_b = raw_lines[i + 1]

            speaker_a, msg_a = self._split_speaker_and_text(raw_a)
            speaker_b, msg_b = self._split_speaker_and_text(raw_b)

            # Skip unwanted lines
            if ("called" in msg_a) or ("called" in msg_b):
                continue
            if ("You missed a call" in msg_a) or ("You missed a call" in msg_b):
                continue
            if ("missed your call." in msg_a) or ("missed your call." in msg_b):
                continue
            if ("sent an attachment." in msg_a) or ("sent an attachment." in msg_b):
                continue

            if speaker_a is None or speaker_b is None:
                continue

            # Determine which line is assistant/user
            is_a_assistant = (speaker_a.lower() == ASSISTANT_NAME) if speaker_a else False
            is_b_assistant = (speaker_b.lower() == ASSISTANT_NAME) if speaker_b else False

            if is_a_assistant and not is_b_assistant:
                assistant_text = msg_a
                user_text = msg_b
            elif is_b_assistant and not is_a_assistant:
                assistant_text = msg_b
                user_text = msg_a
            elif not is_a_assistant and not is_b_assistant:
                # No explicit assistant -> best-effort
                user_text = msg_a
                assistant_text = msg_b
            else:
                # both assistant -> treat first as user, second as assistant
                user_text = msg_a
                assistant_text = msg_b

            # Clean texts
            user_text = self.clean_text(user_text)
            assistant_text = self.clean_text(assistant_text)

            # Format sequence
            prefix = f"{USER_TAG}\n{user_text}\n{ASSISTANT_TAG}\n"
            full_text = prefix + assistant_text

            input_ids = tokenizer.encode(full_text)
            prefix_ids = tokenizer.encode(prefix)

            # Labels: only compute loss for assistant part
            labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]

            if (self.max_length is not None) and (len(input_ids) > self.max_length):
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            self.encoded_text.append((input_ids, labels))
            # file_for_debug.write(full_text + "\n")

    def _split_speaker_and_text(self, line: str):
        if ":" in line:
            name, msg = line.split(":", 1)
            return name.strip(), msg.strip()
        else:
            return None, line.strip()

    def clean_text(self, s: str) -> str:
        # Keep letters and common punctuation
        s = re.sub(r"[^A-Za-z.,!?;:'\"()\[\]{}\-<> ]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def __len__(self):
        return len(self.encoded_text)

    def __getitem__(self, idx):
        return self.encoded_text[idx]  # returns (input_ids, labels)


def custom_collate_fn(
    batch,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    # allowed_max_length: Optional[int] = None,
    device: str = "cpu"
):
    # Shift first
    shifted_inputs, shifted_labels = [], []
    for input_ids, labels in batch:
        if (len(input_ids) <= 3) or (len(labels) <= 3):
            continue  # skip too short sequences
        
        shifted_inputs.append(input_ids[:-1])
        shifted_labels.append(labels[1:])

    # Compute batch max length after shift
    batch_max_length = max(len(seq) for seq in shifted_inputs)

    input_list, label_list = [], []
    for inp, lab in zip(shifted_inputs, shifted_labels):
        inp_padded = inp + [pad_token_id] * (batch_max_length - len(inp))
        lab_padded = lab + [ignore_index] * (batch_max_length - len(lab))
        input_list.append(torch.tensor(inp_padded, dtype=torch.long))
        label_list.append(torch.tensor(lab_padded, dtype=torch.long))

    inputs_tensor = torch.stack(input_list).to(device)
    labels_tensor = torch.stack(label_list).to(device)
    return inputs_tensor, labels_tensor


class ChatDataModule(pl.LightningDataModule):
    """Lightning DataModule for chat datasets with assistant-only training."""

    def __init__(
        self,
        tokenizer,
        train_data: List[str],
        val_data: Optional[List[str]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pad_token_id: int = 50256,
        ignore_index: int = -100,
        max_length: Optional[int] = None,
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
        self.train_dataset = ChatDataset(self.train_data, self.tokenizer, max_length=self.max_length)
        self.val_dataset = ChatDataset(self.val_data, self.tokenizer, max_length=self.max_length)

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
                # allowed_max_length=self.max_length,
                device=self.device,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(
                custom_collate_fn,
                pad_token_id=self.pad_token_id,
                ignore_index=self.ignore_index,
                # allowed_max_length=self.max_length,
                device=self.device,
            ),
        )
