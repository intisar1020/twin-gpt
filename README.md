# twin-gpt
a personal fun project where I  train gpt-2-small model  to communicate like me.

# Getting Started
```bash
git clone https://github.com/intisar1020/twin-gpt.git
pip install poetry
poetry install
```
thats it!.


# Train the model
```bash
python train.py
```

Set your hyperparameters in `./configs/litgpt2.json` file. For a resonable model quality, I recommend to start with a pretrained gpt-2 model checkpoint. You can download the pretrained gpt-2-small checkpoint from hugging face.

# Train dataset format
The training dataset should be in a text file format. Each conversation should be separated by a newline character. Each message in the conversation should be as follows:
```
PersonA: Hello, how are you?
PersonB: I'm good, thank you! How about you?
```

# Inference
```bash
python inference.py --ckpt_path path/to/your/checkpoint.ckpt
```

