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
    existing_ds = load_dataset(base_repo, split="train")
    existing_ds = existing_ds.filter(lambda x: x["language"] == "hinglish")
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
        # messages.append({"role": "assistant", "content": row["answer"]})

        instruction = ""
        for r in messages:
            instruction += "\n\n" + r["content"]

        instruction = createConstraintsPrompt(instruction)
        msg_list = [{"role": "user", "content": instruction}]

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
