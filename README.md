# twin-gpt
a personal fun project where I  train gpt-2-small model  to communicate like me.

# Getting Started
```bash
git clone https://github.com/intisar1020/twin-gpt.git
pip install poetry
poetry install
```
thats it!.

Now activate the poetry shell:
```bash
poetry shell
```


# Train the model
```bash
python train.py
```

Please set your hyperparameters in `./configs/litgpt2.json` file. For a resonable model quality, I recommend to start with a pretrained gpt-2 model checkpoint. You can download the pretrained gpt-2-small checkpoint from hugging face.

I have trained my model on a single NVIDIA RTX 3060 GPU with 12GB memory. You may need to adjust the batch size according to your GPU memory capacity. The higher the batch size I could afford was only 2. Going beyound that will result in out-of-memory error.

# Train dataset format
The training dataset should be in a text file format. Each conversation should be separated by a newline character. Each message in the conversation should be as follows:
```
PersonA: Hello, how are you?
PersonB: I'm good, thank you! How about you?
```

In the train dataloader we mask the loss for PersonA's messages, so that the model only learns to generate PersonB's responses. The rest is similar where we right shift the input tokens by one position to create the target tokens. The objective is next token prediction.


# Inference

To run inference with the trained model, use the following command

```bash
python inference.py --ckpt_path path/to/your/checkpoint.ckpt
```

Please manually set the model path inside the script. The generation hyperparameters can also be set inside the script.

