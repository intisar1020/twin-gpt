from twin_gpt.models.lit_gpt import LitGPT
from scripts.generate import generate
import tiktoken
import torch

import argparse

parser = argparse.ArgumentParser(description="Chat with the model")
parser.add_argument("--ckpt_path", type=str, default="logs/checkpoints/twin_gpt_final.ckpt", help="Path to the model checkpoint")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# text to token
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor

# load the pth for LitGPT
def get_model(ckpt_path, device):
    model = LitGPT.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    return model

model = get_model(args.ckpt_path, device)
print(model)

def chat():
    tokenizer = tiktoken.get_encoding("gpt2")
    print (model.hparams)
    context_size = 1024
    max_new_tokens = 20
    temperature = 0.8
    top_k = 20
    eos_id = tokenizer.encode("\n")[0]  # Assuming the end-of-sequence token is a newline character

    print("You can start chatting with the model (type 'exit' to stop):")
    while True:
        user_input = input("<|user|>: ")
        if user_input.lower() == 'exit':
            break

        input_ids = text_to_token_ids(user_input, tokenizer)
        input_ids = input_ids.to(device)
        output_ids = generate(model, input_ids, max_new_tokens, context_size, temperature, top_k, eos_id)
        output_text = tokenizer.decode(output_ids[0].tolist())
        output_text = output_text.replace(user_input, "")
        print(f"Model: {output_text}")

chat()