from tqdm import tqdm
import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
import torch


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts):
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=512,
        stop=["<|im_end|>"],
    )
    # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
    generations = model.generate(prompts, sampling_params)

    prompt_to_output = {
        g.prompt: g.outputs[0].text.strip() for g in generations
    }
    outputs = [prompt_to_output[prompt]
               if prompt in prompt_to_output else "" for prompt in prompts]

    return outputs


def main(args):

    base_repo = "manishiitg/aditi-dpo-prompts"
    dataset = load_dataset(base_repo, split="train")
    dataset = dataset.filter(lambda x: x["language"] == args.lang)
    dataset = dataset.select(range(10))

    final_data = []

    for row in dataset:
        processed_by = row["processed_by"]
        if args.model_name_or_path not in processed_by:
            final_data.append(row)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.awq:
        print("Loading model and tokenizer vllm awq...")
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            quantization="AWQ",
            max_model_len=8196,
        )
    else:
        print("Loading model and tokenizer vllm...")
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=8196,
        )

    default_system_en = "You are a helpful assistant."
    default_system_hi = "आप एक सहायक सहायक हैं."

    prompts = []
    pending_data = []
    for row in tqdm(final_data):

        prompt = row["prompt"]
        if args.lang == "hi":
            messages = [
                {"role": "system", "content": default_system_hi}
            ]
        else:
            messages = [
                {"role": "user", "content": default_system_en}
            ]

        messages.append(
            {"role": "user", "content": prompt}
        )
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)
        pending_data.append({})

    outputs = eval_hf_model(args, model, tokenizer, prompts)

    for idx, text in enumerate(outputs):
        print("======")
        print("prompt", prompts[idx], "text", text)
        pending_data[idx] = final_data[idx]
        pending_data[idx]["processed_count"] += 1
        processed_by = pending_data[idx]["processed_by"]
        processed_by[args.model_name_or_path] = True
        ratings = pending_data[idx]["ratings"]
        ratings[args.model_name_or_path] = text

        pass

    existing_data = []
    dataset = load_dataset(base_repo, split="train")
    for r in dataset:
        processed_by = row["processed_by"]
        if args.model_name_or_path not in processed_by:
            pass
        else:
            existing_data.append(row)

    final_data = pending_data + existing_data
    dataset = process_and_update_dataset(final_data)
    dataset.push_to_hub("manishiitg/custom-data-chat", private=False)


def process_and_update_dataset(new_data):
    new_data_formatted = {key: [item[key]
                                for item in new_data] for key in new_data[0].keys()}
    new_dataset_chunk = Dataset.from_dict(new_data_formatted)
    dataset2 = new_dataset_chunk
    return dataset2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="model name"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="lang"
    )
    parser.add_argument(
        "--awq",
        action="store_true",
        help="Load model as awq"
    )
    args = parser.parse_args()
    main(args)
