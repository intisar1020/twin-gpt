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
        self.examples = []  # will hold tuples (input_ids:list[int], labels:list[int])

        # Process pairs of lines as before (0,1), (2,3) ...
        for i in range(0, len(raw_lines) - 1, 2):
            raw_a = raw_lines[i]
            raw_b = raw_lines[i + 1]

            speaker_a, msg_a = self._split_speaker_and_text(raw_a)
            speaker_b, msg_b = self._split_speaker_and_text(raw_b)

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

            # Tokenize prefix to find assistant token start
            prefix_ids = tokenizer.encode(prefix)
            full_ids = tokenizer.encode(full_text)

            # Optionally truncate if too long
            if self.max_length is not None and len(full_ids) > self.max_length:
                # keep last max_length tokens (so assistant part is retained preferentially)
                full_ids = full_ids[-self.max_length:]
                # recompute prefix length in tokens (approximate): find first occurrence of prefix end if possible
                # easiest approach: if prefix longer than max_length, we won't have assistant tokens -> skip
                if len(full_ids) <= len(prefix_ids):
                    # skip example (can't train assistant-only target if prefix consumes all tokens)
                    continue
                # adjust prefix_ids length relative to truncated full_ids
                # we assume assistant_token_start = index of first token after prefix in truncated full
                # So compute number of tokens removed from the front:
                # Number removed = original_full_len - new_full_len
                removed = tokenizer.encode(prefix + assistant_text)
                # but to avoid complexity, recompute by re-encoding using the truncated text string:
                # Construct the text back from tokens is not feasible here; so fallback: compute prefix token length as
                assistant_token_start = max(0, len(prefix_ids) - (len(prefix_ids) - (len(full_ids) - len(prefix_ids))))
                # This is approximate; to be robust, prefer not to hit this branch in normal runs.
                assistant_token_start = len(prefix_ids)  # fallback
            else:
                assistant_token_start = len(prefix_ids)

            # Build labels: -100 for tokens before assistant_token_start, actual token ids thereafter
            labels = [-100] * len(full_ids)
            for idx_tok in range(assistant_token_start, len(full_ids)):
                labels[idx_tok] = full_ids[idx_tok]

            # Save example
            # so basically full_ids, and labels are of same length
        
            self.examples.append((full_ids, labels))

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
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]  # (input_ids:list[int], labels:list[int])


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    batch: list of tuples (input_ids:list[int], labels:list[int])
    Returns:
      inputs_tensor: LongTensor (batch_size, seq_len)
      labels_tensor: LongTensor (batch_size, seq_len) with -100 where loss should be ignored
    """
    # Determine max length in batch (add 1 for potential eos)
    batch_max_length = max(len(inp) for inp, _ in batch) + 1

    inputs_lst = []
    labels_lst = []

    for input_ids, labels in batch:
        # Append an <|endoftext|> token to sequence (if desired)
        new_input = list(input_ids) + [pad_token_id]
        new_labels = list(labels) + [ignore_index]  # don't predict eos

        # Pad to batch_max_length
        pad_len = batch_max_length - len(new_input)
        if pad_len > 0:
            new_input = new_input + [pad_token_id] * pad_len
            new_labels = new_labels + [ignore_index] * pad_len
        else:
            new_input = new_input[:batch_max_length]
            new_labels = new_labels[:batch_max_length]

        # Optionally truncate to allowed_max_length
        if allowed_max_length is not None:
            new_input = new_input[:allowed_max_length]
            new_labels = new_labels[:allowed_max_length]

        inputs_lst.append(torch.tensor(new_input, dtype=torch.long))
        labels_lst.append(torch.tensor(new_labels, dtype=torch.long))

    inputs_tensor = torch.stack(inputs_lst)
    labels_tensor = torch.stack(labels_lst)

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