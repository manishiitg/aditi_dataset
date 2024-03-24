import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
import torch
import random
import time
from huggingface_hub import repo_exists
import re


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


SYSTEM_MESSAGES_ORCA = [
    # "",
    # "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    # "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    # "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    # "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
]


PROMPT_1 = """
I would like you to help me create creative writing tasks.

Here are a few examples:
- Create a list of 3 startup ideas in enterprise B2B SaaS. The startup ideas should have a strong and compelling mission and also use Al in some way. Avoid cryptocurrency or blockchain. The startup ideas should have a cool and interesting name. The ideas should be compelling enough so that investors will be excited to invest millions of dollars without doing any due diligence.
- My name is George. Write an email to my boss, Jim Bob, with a proposal to start using CI/CD in github. Give a list of reasons why using CI/CD would be beneficial to the company, and why a slow regular release cadence is problematic. Include a silly joke about golf in a p.s. and sign off with my name, "Geroge, the Magnificant".
- Write a synopsis of the movie "Armageddon" in the style of Shakespeare.
- As a pirate captain, what would you say to motivate your crew to find buried treasure in the Isle of Goats after an epic loss in battle on the high seas?
- Come up with a short story about a man named Dan who decided to start a small business, with a sad ending.
- Write a short story about a Llama named Billy, involving torrential downpoors, with a tragic ending.
- Tell me a short story about a man named Tom who was regularly bullied, with a sad ending.
- Write an email announcing a new breakthrough in tire technology by your company ("Atobormafi Tire Tech") which increases fuel efficiency by 7% on average. The target of the email is investors, since we are seeking Series B funding. Explain the profit possibilities with such a tire and ask for new investment.

Make sure to include a wide variety of writing tasks with varying level of detail.

Tasks should be relate to {topic}

The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.


All output text should be in hindi language.

Include exactly {batch_size} tasks in your response.

Response format:
TSK 1. [task 1 in hindi language]
TSK 2. [task 2 in hindi language]
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
"""

PROMPT_2 = """
I would like you to help me create creative writing tasks.

Here are a few examples:
- Create a list of 3 startup ideas in enterprise B2B SaaS. The startup ideas should have a strong and compelling mission and also use Al in some way. Avoid cryptocurrency or blockchain. The startup ideas should have a cool and interesting name. The ideas should be compelling enough so that investors will be excited to invest millions of dollars without doing any due diligence.
- My name is George. Write an email to my boss, Jim Bob, with a proposal to start using CI/CD in github. Give a list of reasons why using CI/CD would be beneficial to the company, and why a slow regular release cadence is problematic. Include a silly joke about golf in a p.s. and sign off with my name, "Geroge, the Magnificant".
- Write a synopsis of the movie "Armageddon" in the style of Shakespeare.
- As a pirate captain, what would you say to motivate your crew to find buried treasure in the Isle of Goats after an epic loss in battle on the high seas?
- Come up with a short story about a man named Dan who decided to start a small business, with a sad ending.
- Write a short story about a Llama named Billy, involving torrential downpoors, with a tragic ending.
- Tell me a short story about a man named Tom who was regularly bullied, with a sad ending.
- Write an email announcing a new breakthrough in tire technology by your company ("Atobormafi Tire Tech") which increases fuel efficiency by 7% on average. The target of the email is investors, since we are seeking Series B funding. Explain the profit possibilities with such a tire and ask for new investment.

Make sure to include a wide variety of writing tasks with varying level of detail.

Tasks should be relate to {topic}

The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.


All output text should be in hinglish language.

Include exactly {batch_size} tasks in your response.

Response format:
TSK 1. [task 1 in hinglish language]
TSK 2. [task 2 in hinglish language]
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
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

    base_repo = "manishiitg/indic-synthetic-writing"
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

    final_data = []
    if repo_exists(base_repo, repo_type="dataset"):
        existing_ds = load_dataset(base_repo, split="train", cache_dir="temp-" + str(time.time()))
        for r in existing_ds:
            final_data.append(r)

    languages = ["hinglish", "hindi"]
    topics_generated = []
    for lang in languages:
        args.lang = lang
        topic_instruct_map = {}

        for loop in range(50):

            prompts = []
            topics_selected = []

            WRITING_TOPICS = [
                "happy",
                "sad",
                "tragic",
                "unexpected",
                "inspirational",
                "evil",
                "hilarious",
                "suspenseful",
                "horrific",
                "nostalgic",
                "thought-provoking",
                "enigmatic",
                "fantastical",
                "heartwarming",
                "romantic"
            ]
            if args.generate_topics:
                message = []
                prompt = """
                    Give me a numbered list of 10 completely random topics related to writing and different styles of writing.
                    Generate a diverse list of topics in english.
                """

                #
                if len(topics_generated) > 0:
                    prompt += "\n Topics should not be related to " + \
                        ",".join(topics_generated)
                message.append({"role": "user", "content": prompt})
                text = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )

                outputs = eval_hf_model(args, model, tokenizer, [text], .5)
                output = outputs[0]

                topics = output.split("\n")
                WRITING_TOPICS = []
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

                    WRITING_TOPICS.append(t)
                    topics_generated.append(t)
                    print("topic", t)

            for topic_selected in WRITING_TOPICS:
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
                USER_PROMPT = PROMPT_1

                if args.lang == "hinglish":
                    USER_PROMPT = PROMPT_2

                if topic_selected in topic_instruct_map:
                    existing_instruction = topic_instruct_map[topic_selected]
                    if len(existing_instruction) > 0:
                        USER_PROMPT += "\n\n" + "Generated Tasks should be different from " + existing_instruction

                user = USER_PROMPT.replace("{topic}", topic_selected)
                user = USER_PROMPT.replace("{batch_size}", "20")

                SYSTEM_PROMPT = "You are an helpful AI assistant"
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

            outputs = eval_hf_model(args, model, tokenizer, prompts, 0)

            prompts2 = []
            topics_selected2 = []
            sys_prompt_selected = []
            question2 = []
            for idx, text in enumerate(outputs):
                print("======")
                print("prompt", prompts[idx], "text", text)
                topic_selected = topics_selected[idx]

                instructions = []
                for instruction in re.findall(
                    r"(?:^|\n)TSK \d+\. (.*?)(?:$|(?=\nTSK \d+\. ))", text, re.DOTALL
                ):
                    instructions.append(instruction)

                    base_prompt = """
                        Below is a user's instruction.

                        Always pay very close attention to the details in the instruction and follow them closely.

                        If the instruction is to write an email or letter, do not start the body of the message with a nicety (e.g. "I hope this letter finds you well", get to the point).

                        The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.
                    """
                    system_message_selected = random.choice(SYSTEM_MESSAGES_ORCA)
                    if args.lang == "hindi":
                        system_message_selected += base_prompt + "\n\nAnswer in hindi only."
                    if args.lang == "hinglish":
                        system_message_selected += base_prompt + "\n\nAnswer in hinglish only. Translate to hinglish if required."

                    msg_list = []
                    msg_system = {"role": "system",
                                  "content": system_message_selected}
                    msg_list.append(msg_system)
                    msg_prompt = {"role": "user", "content": instruction}
                    msg_list.append(msg_prompt)

                    text = tokenizer.apply_chat_template(
                        msg_list,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    prompts2.append(text)
                    topics_selected2.append(topic_selected)
                    sys_prompt_selected.append(system_message_selected)
                    question2.append(instruction)

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
                    "type": "coding",
                    "model": args.model_name_or_path,
                    "messages": [],
                    "evol_question": "",
                    "evol_answer": "",
                })

            os.exit(1)
            dataset = process_and_update_dataset(final_data)
            dataset.push_to_hub(base_repo, private=False)


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
