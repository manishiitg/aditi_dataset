from tqdm import tqdm
import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
import time
from datasets import Dataset
import torch
import random
import re

import unicodedata


def is_hindi(char):
    try:
        return unicodedata.name(char).startswith('DEVANAGARI')
    except ValueError:
        return False


def contains_hindi(s):
    return any(is_hindi(char) for char in s)


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


base_instruction = "I want you act as a Prompt Rewriter.\r\n \
					Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
					But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"


def createConstraintsPrompt(instruction):
    prompt = base_instruction.format(
        "Please add one more constraints/requirements into #The Given Prompt#'")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createDeepenPrompt(instruction):
    prompt = base_instruction.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createConcretizingPrompt(instruction):
    prompt = base_instruction.format(
        "Please replace general concepts with more specific concepts.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createReasoningPrompt(instruction):
    prompt = base_instruction.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


base_instruction_breadth = "I want you act as a Prompt Creator.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"


def createBreadthPrompt(instruction):
    prompt = base_instruction_breadth
    prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Created Prompt#:\r\n"
    return prompt


def main(args):

    base_repo = "manishiitg/indic-synthetic-code"

    max_rows = 1000
    final_data = []
    existing_ds = load_dataset(base_repo, split="train", cache_dir="temp-" + str(time.time()))
    # existing_ds = existing_ds.shuffle()
    for r in existing_ds:
        if "messages" not in r:
            r["messages"] = []
            r["evol_question"] = ""
            r["eval_answer"] = ""
        # is_hindi = contains_hindi(r["question"])
        if len(final_data) < max_rows and len(r["evol_question"]) == 0:
            final_data.append(r)

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

        # evol instruct
        prompts = []
        for row in final_data:
            instruction = row["question"]
            evol_prompts = []
            evol_prompts.append(createConstraintsPrompt(instruction))
            evol_prompts.append(createDeepenPrompt(instruction))
            evol_prompts.append(createConcretizingPrompt(instruction))
            evol_prompts.append(createReasoningPrompt(instruction))
            evol_prompts.append(createBreadthPrompt(instruction))

            selected_evol_prompt = random.choice(evol_prompts)
            if row["language"] == "hindi":
                selected_evol_prompt += "\n\nAnswer in hindi only"
            if row["language"] == "hinglish":
                selected_evol_prompt += "\n\nAnswer in hinglish only. Translate to hinglish if required"

            messages = []
            messages.append({"role": "user", "content": selected_evol_prompt})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            prompts.append(text)

        outputs = eval_hf_model(args, model, tokenizer, prompts, 0)

        prompts2 = []
        questions = []
        for idx, text in enumerate(outputs):
            questions.append(text)

            messages = []
            messages.append(
                {"role": "system", "content": row["system_prompt"]})
            messages.append({"role": "user", "content": text})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            prompts2.append(text)

        outputs2 = eval_hf_model(args, model, tokenizer, prompts2, .2)
        for idx, text in enumerate(outputs2):
            print("======")
            print("eval prompt", questions[idx], "text", text)
            final_data[idx]["evol_question"] = questions[idx]
            final_data[idx]["evol_answer"] = text

        final_data_hash = {}
        for r in final_data:
            hash = r["question"] + r["answer"]
            final_data_hash[hash] = True

        existing_ds = load_dataset(base_repo, split="train", cache_dir="temp-" + str(time.time()))
        existing_data = []
        for r in existing_ds:
            hash = r["question"] + r["answer"]

            if hash not in final_data_hash:
                existing_data.append(r)

        existing_data = final_data + existing_data
        dataset = process_and_update_dataset(existing_data)
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
        "--awq",
        action="store_true",
        help="Load model as awq"
    )
    args = parser.parse_args()
    main(args)
