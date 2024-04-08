from tqdm import tqdm
import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
import torch
import time


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts):
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=2048,
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
    dataset = dataset.filter(lambda x: x["is_repeating"])
    if args.lang == "hinglish":
        dataset = dataset.filter(lambda x: x["language"] == "en").shuffle()

    final_data = []
    max_rows = 5

    if args.model_name_or_path == "Qwen/Qwen1.5-72B-Chat-AWQ":
        if args.lang == "hinglish":
            key = "chosen2"
        else:
            key = "chosen3"
    else:
        key = "rejected"

    for row in dataset:

        # if len(row[key]) == 0 and len(final_data) < max_rows:
        #     if args.lang == "hinglish":
        #         if len(row["prompt"]) < 250:
        #             continue
        final_data.append(row)

    if len(final_data) > 0:
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

        default_system_en = "You are a helpful assistant. Provide a short, accurate and well formatted response."

        prompts = []
        if args.lang == "hinglish":
            for row in tqdm(final_data):

                prompt = row["prompt"]
                messages = [
                    {"role": "user", "content": "Translate given text to hinglish language. Only translate given input, don't do anything else."}
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

            outputs = eval_hf_model(args, model, tokenizer, prompts)

        prompts = []
        hinglish_prompts = []
        for idx, row in tqdm(enumerate(final_data)):
            if args.lang == "hinglish":
                prompt = outputs[idx]
                hinglish_prompts.append(prompt)
                print("--------")
                print("prompt", prompt)
                print("org prompt", row["prompt"])
            else:
                prompt = row["prompt"]
                print("prompt", prompt)

            if args.lang == "hi":
                messages = [
                    {"role": "system", "content": default_system_en + "If users question in related to programing, always insert inline comments. Reply only in hindi language."}
                ]
            elif args.lang == "hinglish":
                messages = [
                    {"role": "system", "content": default_system_en + "If users question in related to programing, always insert inline comments. Reply only in hinglish language."}
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

        outputs = eval_hf_model(args, model, tokenizer, prompts)

        uuid_row_map = {}
        for idx, text in enumerate(outputs):
            print("======")
            print("prompt", prompts[idx], "text", text)

            uuid = final_data[idx]["uuid"]
            final_data[idx][key] = text
            uuid_row_map[uuid] = final_data[idx]

        existing_data = []
        dataset = load_dataset(base_repo, split="train",
                               cache_dir="temp-" + str(time.time()))
        for row in dataset:
            uuid = row["uuid"]
            if uuid in uuid_row_map:
                processed_row = uuid_row_map[uuid]
                existing_data.append(processed_row)
            else:
                existing_data.append(row)

        final_data = existing_data
        dataset = process_and_update_dataset(final_data)
        dataset.push_to_hub(base_repo, private=True)


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
