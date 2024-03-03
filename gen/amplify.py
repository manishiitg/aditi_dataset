from tqdm import tqdm
import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
import torch
import random
import re


def createDeepenPrompt(language):
    prompt = """
    Based on the conversion between an ai assistant and user, generate next possible question which a user can ask.
    In the conversion "user:" means the question as by user and "assistant:" means answer given.

    Your task to generate a next possible question which user can ask based on the conversation.
    If the given conversation contains inquiries about certain issues, the depth and breadth of the inquiry can be increased
    Reply only with question generated.
    """
    if language == "hinglish":
        prompt += "\n Generate question in hinglish only."
    return prompt


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts, temperature):
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
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

    base_repo = "manishiitg/indic-synthetic-instruct"

    max_rows = 5
    final_data = []
    args.language = "hinglish"
    existing_ds = load_dataset(base_repo, split="train")
    existing_ds = existing_ds.filter(lambda x: x["language"] == args.language)
    for r in existing_ds:
        if len(final_data) < max_rows:
            final_data.append(r)

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

    prompts = []
    for row in final_data:

        messages = []
        # messages.append({"role": "system", "content": row["system_prompt"]})
        messages.append({"role": "user", "content": row["question"]})
        messages.append({"role": "assistant", "content": row["answer"]})

        instruction = ""
        for r in messages:
            instruction += r["role"] + ":" + r["content"] + "\n\n"

        system = createDeepenPrompt(args.language)
        msg_list = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction}
        ]

        text = tokenizer.apply_chat_template(
            msg_list,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)

    outputs = eval_hf_model(args, model, tokenizer, prompts, 0)

    prompts2 = []
    topics_selected2 = []
    sys_prompt_selected = []
    question2 = []
    for idx, text in enumerate(outputs):
        print("======")
        print("prompt", prompts[idx], "text", text)

        # Define the regex pattern to match the instructions
        # instruction_pattern = r'\*([^*]*)\*'
        # Find all matches for instructions
        # instructions = re.findall(
        #     instruction_pattern, text, re.DOTALL)

        # instructions = []
        # matches = text.split("\n")
        # for match in matches:
        #     if "." in match:
        #         ix = match.index(".")
        #         match = match[ix+1:]
        #     else:
        #         print("skipping instruction", match)
        #         continue
        #     match = match.strip()
        #     if match.startswith('"'):
        #         match = match[0:]
        #     if match.endswith('"'):
        #         match = match[:-1]
        #     instructions.append(match.strip())

    # outputs2 = eval_hf_model(args, model, tokenizer, prompts2, .1)
    # for idx, text in enumerate(outputs2):
    #     print("======")

    #     print("topic selected", topics_selected2[idx])
    #     print("text", question2[idx])
    #     print("text", text)
    #     final_data.append({
    #         "topic": topics_selected2[idx],
    #         "question": question2[idx],
    #         "answer": text,
    #         "system_prompt": sys_prompt_selected[idx],
    #         "language": args.lang,
    #         "type": "alpaca",
    #         "model": args.model_name_or_path,
    #     })

    # dataset = process_and_update_dataset(final_data)
    # dataset.push_to_hub(base_repo, private=True)


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
        "--awq",
        action="store_true",
        help="Load model as awq"
    )
    args = parser.parse_args()
    main(args)
