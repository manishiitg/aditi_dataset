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
        max_tokens=4096,
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
    dataset = load_dataset(base_repo, split="train",
                           cache_dir="temp-" + str(time.time()))

    final_data = []
    max_rows = 100
    # required because will be running in a distributed way

    push_data = []
    for row in dataset:
        if row["language"] == args.lang:
            processed_by = row["processed_by"]
            if args.model_name_or_path not in processed_by and len(final_data) < max_rows:
                final_data.append(row)
                row["processed_by"][args.model_name_or_path] = True
            elif args.model_name_or_path in processed_by and not processed_by[args.model_name_or_path] and len(final_data) < max_rows:
                final_data.append(row)
                row["processed_by"][args.model_name_or_path] = True

            if args.model_name_or_path not in row["processed_by"]:
                row["processed_by"][args.model_name_or_path] = False
            else:
                if row["responses"][args.model_name_or_path]:
                    row["processed_by"][args.model_name_or_path] = True
                else:
                    row["processed_by"][args.model_name_or_path] = False

        push_data.append(row)

    dataset = process_and_update_dataset(push_data)
    dataset.push_to_hub(base_repo, private=True)

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
                max_model_len=8196*2,
            )
        else:
            print("Loading model and tokenizer vllm...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.model_name_or_path,
                tokenizer_mode="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=8196*2,
            )

        default_system_en = "You are a helpful assistant."
        default_system_hi = "आप एक सहायक सहायक हैं."

        prompts = []
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

        outputs = eval_hf_model(args, model, tokenizer, prompts)

        uuid_row_map = {}
        for idx, text in enumerate(outputs):
            print("======")
            print("prompt", prompts[idx], "text", text)

            uuid = final_data[idx]["uuid"]
            final_data[idx]["processed_count"] += 1
            processed_by = final_data[idx]["processed_by"]
            processed_by[args.model_name_or_path] = True
            final_data[idx]["responses"][args.model_name_or_path] = text
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
