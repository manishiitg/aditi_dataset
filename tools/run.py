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


USER_PROMPT = """
I would like you to help me generate prompts for a large language model to help train it in functional calling.

To accomplish this, I want you to generate 3 to 6 tools similar to below.

Example of tools:
[
  {
    "name": "search_recipe",
    "description": "Search for a recipe based on ingredients",
    "parameters": {
      "type": "object",
      "properties": {
        "ingredients": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "The ingredients available"
        },
        "cuisine": {
          "type": "string",
          "description": "The cuisine type"
        },
        "diet": {
          "type": "string",
          "description": "The dietary restrictions"
        },
        "time": {
          "type": "integer",
          "description": "The maximum cooking time in minutes"
        }
      },
      "required": [
        "ingredients"
      ]
    }
  },
  {
    "name": "convert_currency",
    "description": "Convert one currency to another",
    "parameters": {
      "type": "object",
      "properties": {
        "amount": {
          "type": "number",
          "description": "The amount to convert"
        },
        "from_currency": {
          "type": "string",
          "description": "The currency to convert from"
        },
        "to_currency": {
          "type": "string",
          "description": "The currency to convert to"
        }
      },
      "required": [
        "amount",
        "from_currency",
        "to_currency"
      ]
    }
  }
]


The tools generated should be related to related to:
{topic_selected}

Tools generated should be in json format with "name", "description", "parameters", "required" as mandatory field.
Be sure to format the list of available functions as proper JSON, with appropriate spacing for nested objects.

"""

PROMPT1_QUESTION_GEN = """
TOOLS:
{context}

Your task is to generate complex task(s) that asks the user to generate a response based exclusively on the information of one or more of the generated tools blocks.

The task(s) should be questions or instructions. The task(s) should not specifically indicate that the user should reference the tools, just state the task(s).

Do not include phrasing such as "Using the first text block", or "using the blog post", etc., just assume the target audience will know where to find the answer based on the question/instruction.

The task(s) must not start with "Describe the ...", "Explain how ...", etc., and should ask for specific information, and must be completely and accurately answerable using only the random text.

Ask TRICKY questions, so that the user really needs to think before he is able to answer. 

You also need to generate question another set of questions, which might CONFUSE the agent and force him to hallucinate. 

1. Generate diverse questions which would require the tools to be used with values for all the required parameters.
2. Generate random questions which require tools to be used but with few required parameters missing. Don't specially mention parameters are missing in the question.
3. Generate questions which don't require tools be used at all.

When generating questions, don't mention the word "TOOLS" in the questions.

Generate tasks in {language} language only.

Respond in the following format.
List of 5 tasks generated with tools and parameters:
TSK 1. [task 1 in {language} language]
TSK 2. [task 2 in {language} language]

List of 5 tasks generated with parameters missing.
TSK 1. [task 1 in {language} language]
TSK 2. [task 2 in {language} language]

List of 5 tasks generated which don't require tools.
TSK 1. [task 1 in {language} language]
TSK 2. [task 2 in {language} language]
"""


PROMPT1_RESPONSE = """
TOOLS
{context}

You are given a TOOLS defination.
You need to generate a relevent output based on users question using TOOLS in json format.

Rules For Using TOOLS:
    You ONLY have access to TOOLS listed above, and should NEVER make up tools that are not listed here.
    If a value for tool parameter is missing, don't make assumptions about the value always ask the user.
    You can only use a tool, when you have all parameter values. If you don't have values for all parameters, return "no_tools"
    Always ask user about missing parameter values.
    If there is no tool which fits the, reply with "no_tools"
    
    If you are selecting a TOOL, reply in the exact format
    {'arguments': <args-dict>, 'name': <function-name>}

If executing  a TOOL:
Make sure you have all required parameter values as per the parameter types.
Do not make up parameter values, make sure you have exact value as per the required parameter type.

Reply in format below:

Thought in English: think step by step about what to do in detail.
Action: the action to take if you have all tool parameter values, only one name of tool_names, in the exact format {'arguments': <args-dict>, 'name': <function-name>}
Should Execute Action: do we have all parameter values to execute action reply only yes or no. If yes, i.e executing a tool reply to user should only contain a message asking user to wait.
Reply to User: a short natural language based message to be sent to the user
"""


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts, temperature, max_tokens=8196):
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
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

    base_repo = "manishiitg/indic-synthetic-tools"
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

    languages = ["hinglish"]
    topic_selected = "tools"
    PROGRAMMING_TOPICS = []
    for lang in languages:
        args.lang = lang
        topics_generated = []
        if lang in topics_generated_map:
            topics_generated = topics_generated_map[lang]
        topic_instruct_map = {}

        for loop in range(5):
            prompts = []
            if args.generate_topics or True:
                message = []
                prompt = """Give me a numbered list of 50 completely random topics which could be related to popular open source API's available online related to specific tasks/websites.
                 Every topic should be in a new line."""
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

            topics = []
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

                user = USER_PROMPT.replace("{topic_selected}", topic_selected)
                user = user.replace("{language}", lang)
                SYSTEM_PROMPT = "You are an helpful AI assistant"
                msg_system = {"role": "system", "content": SYSTEM_PROMPT}
                msg_list.append(msg_system)
                msg_prompt = {"role": "user", "content": user}
                msg_list.append(msg_prompt)

                text = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
                topics.append(topic_selected)

            outputs = eval_hf_model(args, model, tokenizer, prompts, 0)

            prompts = []
            contexts = []
            for idx, context in enumerate(outputs):
                user = PROMPT1_QUESTION_GEN.replace("{context}", context)
                user = user.replace("{task_count}", "5")
                user = user.replace("{language}", lang)

                msg_list = []
                SYSTEM_PROMPT = "You are an helpful AI assistant"
                if lang == "hinglish":
                    SYSTEM_PROMPT += "\n Reply in hinglish only."

                msg_system = {"role": "system", "content": SYSTEM_PROMPT}
                msg_list.append(msg_system)
                msg_prompt = {"role": "user", "content": user}
                msg_list.append(msg_prompt)

                text = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
                contexts.append(context)

            max_tokens = 2048
            if lang == "hindi":
                max_tokens = 4096
            outputs = eval_hf_model(args, model, tokenizer, prompts, .2, max_tokens)

            prompts2 = []
            contexts2 = []
            topics2 = []
            global_questions = []
            for idx, questions_text in enumerate(outputs):
                context = contexts[idx]

                for instruction in re.findall(
                    r"(?:^|\n)TSK \d+\. (.*?)(?:$|(?=\nTSK \d+\. ))", questions_text, re.DOTALL
                ):
                    
                    if "List of 5 tasks" in instruction:
                        ixxx = instruction.find("List of 5 tasks")
                        instruction = instruction[:ixxx]

                    print("instruction", instruction)

                    user = PROMPT1_RESPONSE.replace("{language}", lang)
                    user = user.replace("{context}", context)

                    msg_list = []
                    msg_system = {"role": "system", "content": user}
                    msg_list.append(msg_system)
                    msg_prompt = {"role": "user", "content": instruction}
                    msg_list.append(msg_prompt)

                    contexts2.append(contexts[idx])
                    global_questions.append(instruction)
                    topics2.append(topics[idx])

                    text = tokenizer.apply_chat_template(
                        msg_list,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts2.append(text)

            max_tokens = 1024
            outputs = eval_hf_model(args, model, tokenizer, prompts2, 0, max_tokens)
            for idx, text in enumerate(outputs):
                # print("context", contexts2[idx])
                print("question", global_questions[idx])
                print("answer", text)
                print("========")

                final_data.append({
                    "topic": topics2[idx],
                    "question": global_questions[idx],
                    "answer": text,
                    "system_prompt": contexts2[idx],
                    "language": args.lang,
                    "type": "tools",
                    "model": args.model_name_or_path,
                    "messages": [],
                    "evol_question": "",
                    "evol_answer": "",
                })

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
