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


from system_messages import (
    SYSTEM_MESSAGES_ORCA,
    SYSTEM_MESSAGES_TESS,
    SYSTEM_MESSAGES_CODE,
)
from topics import TOPICS

code_data = False

if code_data:
    SYSTEM_MESSAGES = SYSTEM_MESSAGES_CODE
    PROMPT_1 = """For the following SUBJECT_AREA, generate a question that covers a very narrow topic in the SUBJECT_AREA, with sufficient depth and breadth. The topic in the question should be important to the SUBJECT_AREA, with known-answers present. The generated question should be detailed, seek true nature of our universe from first principles, curiosity invoking, thought provoking, and also should be able to be answered by an intelligence like yourself. Make sure the question is sufficiently harder and multi-part, like a graduate level course question. The question should seek an answer with computer code."""

else:
    SYSTEM_MESSAGES = SYSTEM_MESSAGES_ORCA + SYSTEM_MESSAGES_TESS
    PROMPT_1 = """For the following SUBJECT_AREA, generate a question that covers a very narrow topic in the SUBJECT_AREA, with sufficient depth and breadth. The topic in the question should be important to the SUBJECT_AREA, with known-answers present. The generated question should be detailed, seek true nature of our universe from first principles, curiosity invoking, thought provoking, and also should be able to be answered by an intelligence like yourself. Make sure the question is sufficiently harder and multi-part, like a graduate level course question."""


msg_context = {"role": "system", "content": str(PROMPT_1)}


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts):
    sampling_params = vllm.SamplingParams(
        temperature=.7,
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

    prompts = []
    pending_data = []

    topic_number = random.randint(0, len(TOPICS))
    topic_selected = TOPICS[topic_number]
    system_message_number = random.randint(0, len(SYSTEM_MESSAGES))
    system_message_selected = SYSTEM_MESSAGES[system_message_number]
    system_message_generation = PROMPT_1

    for row in tqdm(range(10)):

        msg_list = []
        msg_system = {"role": "system", "content": PROMPT_1 +
                      "\n Question should be related to india."}
        msg_list.append(msg_system)
        msg_prompt = {"role": "user",
                      "content": f"SUBJECT_AREA: {topic_selected}"}
        msg_list.append(msg_prompt)
        text = tokenizer.apply_chat_template(
            msg_list,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)
        pending_data.append({})

    outputs = eval_hf_model(args, model, tokenizer, prompts)

    for idx, text in enumerate(outputs):
        print("======")
        print("prompt", prompts[idx], "text", text)
        pass


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
