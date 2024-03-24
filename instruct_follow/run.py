import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
import torch
import random
import re
import time
from huggingface_hub import repo_exists


import unicodedata


def is_hindi(char):
    try:
        return unicodedata.name(char).startswith('DEVANAGARI')
    except ValueError:
        return False


def contains_hindi(s):
    return any(is_hindi(char) for char in s)


def contains_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


PROMPT_1 = """
You are asked to come up with a set of 25 diverse task instructions. 
These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

The instruction should only be related to SUBJECT_AREA

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be related to only SUBJECT_AREA
    a. The type of instruction should not include poem writing
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, knowledge based questions, classification, editing, etc.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should involve realistic data and should not contain simple placeholders. The instructions should provide substantial content to make the instruction challenging but should ideally not exceed 2 to 3 sentences.
6. Make sure every instruction is in hindi language only. 

List of 25 tasks:

1. <instruction_in_hindi>
2. <instruction_in_hindi>
"""

PROMPT_2 = """
You are asked to come up with a set of 25 diverse task instructions and corrosponding input. 

You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.

These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions based on the input.

The instruction should only be related to SUBJECT_AREA

Here are the requirements:
1. Generate tasks only in hinglish. Translate tasks generated to hinglish if tasks are in hindi.
2. Try not to repeat the verb for each instruction to maximize diversity.
3. The type of instructions should be diverse. 
    a. The list should include diverse types of tasks like open-ended generation, classification, editing, writing, conversation, trivia, etc.
4. The type of instructions should be related to only SUBJECT_AREA
5. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted
7. The instructions should involve realistic data and should not contain simple placeholders. The instructions should provide substantial content to make the instruction challenging but should ideally not exceed 2 to 3 sentences.

List of 25 tasks:

The output format should be:
INSTRUCTION: [first instruction in hinglish]
INPUT: [first input's answer in hinglish]

INSTRUCTION: [second instruction in hinglish]
INPUT: [second instruction's answer in hinglish]

"""


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

    base_repo = "manishiitg/indic-synthetic-instruct-follow"
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

    final_data = []
    if repo_exists(base_repo, repo_type="dataset"):
        existing_ds = load_dataset(base_repo, split="train", cache_dir="temp-" + str(time.time()))
        for r in existing_ds:
            final_data.append(r)

    global TOPICS

    topics_generated = []

    languages = ["hinglish", "hindi"]
    for lang in languages:
        args.lang = lang
        topic_instruct_map = {}

        for loop in range(10):

            prompts = []
            topics_selected = []
            if args.generate_topics or True:
                message = []
                prompt = """
                    Give me a numbered list of 50 completely random topics
                    Generate a diverse list of topics in english.
                """
                if len(topics_generated) > 0:
                    prompt += "\n Topics should not be related to " + \
                        ",".join(topics_generated)
                message.append({"role": "user", "content": prompt})
                text = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )

                outputs = eval_hf_model(args, model, tokenizer, [text], .25)
                output = outputs[0]

                topics = output.split("\n")
                TOPICS = []
                for t in topics:
                    try:
                        idx = t.index(".")
                        if idx != -1:
                            t = t[idx + 1:]
                            t = t.strip()
                    except ValueError:
                        pass

                    if t.startswith('"'):
                        t = t[1:]
                    if t.endswith('"'):
                        t = t[:-1]

                    TOPICS.append(t)
                    topics_generated.append(t)
                    print("topic", t)

            for topic_selected in TOPICS:

                existing_instructions = []
                for r in final_data:
                    if r["language"] == lang:
                        if r["topic"] == topic_selected:
                            existing_instructions.append(r["question"])

                random.shuffle(existing_instructions)
                if len(existing_instructions) > 25:
                    topic_instruct_map[topic_selected] = ",".join(existing_instructions[:25])
                else:
                    topic_instruct_map[topic_selected] = ",".join(existing_instructions)

                msg_list = []
                SYSTEM_PROMPT = PROMPT_1

                if args.lang == "hinglish":
                    SYSTEM_PROMPT = PROMPT_2

                user = f"SUBJECT_AREA: {topic_selected}"

                if topic_selected in topic_instruct_map:
                    existing_instruction = topic_instruct_map[topic_selected]
                    user += "\n\n" + "Generated Instructions should be different from " + existing_instruction

                msg_system = {"role": "system", "content": SYSTEM_PROMPT}
                msg_list.append(msg_system)
                msg_prompt = {"role": "user",
                              "content": user}
                msg_list.append(msg_prompt)

                text = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
                topics_selected.append(topic_selected)

            outputs = eval_hf_model(args, model, tokenizer, prompts, .2)
            print(outputs)
            

            prompts2 = []
            topics_selected2 = []
            sys_prompt_selected = []
            question2 = []
            for idx, text in enumerate(outputs):
                print("======")

                start_key = "INSTRUCTION"
                end_key = "INPUT"

                pattern = re.compile(f"{start_key}:(.*?){end_key}:(.*?)(?={start_key}|$)", re.DOTALL)

                matches = pattern.findall(text)

                for question, answer in matches:
                    print("======")
                    print(f"INSTRUCTION: {question.strip()}")
                    print(f"INPUT: {answer.strip()}")
                    print()

                    msg_list = []
                    msg_system = {"role": "system",
                                  "content": question.strip()}
                    msg_list.append(msg_system)
                    msg_prompt = {"role": "user", "content": answer.strip()}
                    msg_list.append(msg_prompt)

                    text = tokenizer.apply_chat_template(
                        msg_list,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts2.append(text)
                    topics_selected2.append(topic_selected)
                    sys_prompt_selected.append(question.strip())
                    question2.append(answer.strip())

            os.exit(1)
            outputs2 = eval_hf_model(args, model, tokenizer, prompts2, .1)
            for idx, text in enumerate(outputs2):
                print("======")

                print("topic selected", topics_selected2[idx])
                print("text", question2[idx])
                print("text", text)
                final_data.append({
                    "topic": topics_selected2[idx],
                    "question": question2[idx],
                    "answer": text,
                    "system_prompt": sys_prompt_selected[idx],
                    "language": args.lang,
                    "type": "alpaca",
                    "model": args.model_name_or_path,
                    "messages": [],
                    "evol_question": "",
                    "evol_answer": "",
                })

            dataset = process_and_update_dataset(final_data)
            dataset.push_to_hub(base_repo, private=True)


def process_and_update_dataset(new_data):
    new_data_formatted = {key: [item[key]
                                for item in new_data] for key in new_data[0].keys()}
    new_dataset_chunk = Dataset.from_dict(new_data_formatted)
    dataset2 = new_dataset_chunk
    return dataset2


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
    parser.add_argument(
        "--generate_topics",
        action="store_true",
        help="generate topocs"
    )

    args = parser.parse_args()
    main(args)
