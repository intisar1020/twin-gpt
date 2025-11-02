import json
import torch
import tiktoken
from twin_gpt.models.lit_gpt import LitGPT
from scripts.generate import generate

CKPT_PATH = "logs/checkpoints/twin_gpt_final.ckpt"
INPUT_JSON = "data/instruct.json"
OUTPUT_JSON = "./instructions_output.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 40

def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    prompt = (
        "Below is an instruction that describes a task paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
        f"\n\n### Input:\n{input_text}"
        "\n\n### Response:\n"
    )
    return prompt

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def get_model(ckpt_path, device):
    model = LitGPT.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    return model.to(device)

def main():
    model = get_model(CKPT_PATH, DEVICE)
    tokenizer = tiktoken.get_encoding("gpt2")
    eos_id = tokenizer.encode("\n")[0]
    context_size = getattr(model.hparams, "block_size", 1024)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    results = []
    tot = 0
    for entry in instructions:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        prompt = build_instruction_prompt(instruction, input_text)
        input_ids = text_to_token_ids(prompt, tokenizer).to(DEVICE)
        output_ids = generate(
            model,
            idx=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=context_size,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            eos_id=eos_id,
        )
        output_text = tokenizer.decode(output_ids[0].tolist())
        response_only = output_text[len(prompt):].strip()
        response_only = response_only.split("<|endoftext|>")[0].strip()  # i am manually adding this to stop at end token
        results.append({
            "instruction": instruction,
            "input": input_text,
            "output": response_only
        })
        print(f"Processed instruction: {instruction[:50]}...")
        tot += 1
        if (tot == 10):
            break  

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
