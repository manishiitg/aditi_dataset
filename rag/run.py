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
    "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
]


USER_PROMPT = """
I would like you to help me generate prompts for a large language model to help train it to reduce hallucinations.

To accomplish this, I want you to generate 3 to 8 random text block(s) with random names, numbers, locations, facts, etc., making sure the content of the text does not correlate too closely with known/accurate information.

If the topic is about a specific person, place, or historical event, change the dates, locations, and facts but keep the person/place/event the same. For example, if the text is about Joe Biden, and the text indicates a date of birth of November 20, 1942, the random text should select a new random date for DoB but keep it about Joe Biden (i.e., don't change the name).

The random text block(s) should be extremely realistic, and should not include any placeholders.

Make sure the random text blocks are atleast more than 150 words.

Each text block should be in English, but "BEGININPUT", "ENDINPUT" are special tokens that must not be translated.

Random text block writing style:
The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.

The random text block(s) should be in the style:
- news article
- blog post
- slack conversation
- text messages
- scientific study
- medical report
- reddit post with replies
- email
- tweet
- jira ticket
- github merge request
- gitlab issue
- how-to article

The random text blocks should not reference each other.

Don't mention text block style in generated text.

The facts should be related to:
{topic_selected}

Each text block must be formatted as:
BEGININPUT
[random text goes here]
ENDINPUT

Make sure every text block has the exact formatting specified, including ALL tags "BEGININPUT" and a trailing "ENDINPUT".

Make sure every text block has the exact formatting specified, including ALL tags "BEGININPUT" and a trailing "ENDINPUT".

After generating the text block(s), ensuring details such as dates, places, misc. factoids are randomized, add {task_count} complex task(s) that asks the user to generate a response based exclusively on the information of {target_selection}

The task(s) should be questions or instructions. The task(s) should not specifically indicate that the user should reference the text, just state the task(s).

Do not include phrasing such as "Using the first text block", or "using the blog post", etc., just assume the target audience will know where to find the answer based on the question/instruction.

The task(s) must not start with "Describe the ...", "Explain how ...", etc., and should ask for specific information, and must be completely and accurately answerable using only the random text.

Output format should be:
[list of text blocks in the format described]
BEGININSTRUCTION
[random task(s) go here]
ENDINSTRUCTION
"""

@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts, temperature):
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
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

    base_repo = "manishiitg/indic-synthetic-rag"
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
    topics_generated_map = {}
    if repo_exists(base_repo, repo_type="dataset"):
        existing_ds = load_dataset(base_repo, split="train", cache_dir="temp-" + str(time.time()))
        for r in existing_ds:
            final_data.append(r)
            if r["language"] not in topics_generated_map:
                topics_generated_map[r["language"]] = []
            topics_generated_map[r["language"]].append(r["topic"])

    languages = ["hinglish", "english", "hindi"]
    topic_selected = "rag"
    PROGRAMMING_TOPICS = []
    for lang in languages:
        args.lang = lang
        topics_generated = []
        if lang in topics_generated_map:
            topics_generated = topics_generated_map[lang]
        topic_instruct_map = {}

        for loop in range(20):
            prompts = []
            if args.generate_topics or True:
                message = []
                prompt = """
                    Give me a numbered list of 50 completely random topics related to reasoning.
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
                PROGRAMMING_TOPICS = []
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

                    PROGRAMMING_TOPICS.append(t)
                    topics_generated.append(t)
                    print("topic", t)
            
            for topic_selected in PROGRAMMING_TOPICS:
                existing_instructions = []
                for r in final_data:
                    if r["language"] == lang:
                        if r["topic"] == topic_selected:
                            existing_instructions.append(r["question"])

                random.shuffle(existing_instructions)
                if len(existing_instructions) > 50:
                    topic_instruct_map[topic_selected] = ",".join(existing_instructions[:50])
                else:
                    topic_instruct_map[topic_selected] = ",".join(existing_instructions)

                msg_list = []

                # if topic_selected in topic_instruct_map:
                #     existing_instruction = topic_instruct_map[topic_selected]
                #     if len(existing_instruction) > 0:
                #         USER_PROMPT += "\n\n" + "Generated questions should be different from " + existing_instruction

                user = USER_PROMPT.replace("{batch_size}", "5")
                user = user.replace("{topic_selected}", topic_selected)
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

                outputs = eval_hf_model(args, model, tokenizer, prompts, 0)

                for idx, text in enumerate(outputs):
                    # print("prompt", prompts[idx], "text", text)
                    pass

                    

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