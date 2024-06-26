from tqdm import tqdm
import argparse
import os
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
import torch
import re
import time
from huggingface_hub import repo_exists

# in this we simply save prompts outputs to a huggingface repo
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl

prompt = """
As an impartial evaluator, please assess the quality of the AI assistant's response to the user's question below. 
Your evaluation should take into account several factors, including relevance, accuracy. 

[Question]
{question}
[AI Assistant's Response]
{answer}

Please rate the response on a scale of 1 to 10 for each of the following evaluation factors:

Relevance: The extent to which the response is related to the user's question or topic.
Accuracy: The correctness of the information provided in the response based on the question asked.


Calculate an overall rating based on above factors and also provide an detailed explanation for the overall rating.
Also classify the conversation into a category like coding, general, language, roleplay, writing, wordgame, joke, rp, math
If category is not from above list, generate a category yourself.

Only respond in json format as follows:
{
  "overall_rating": {
    "explanation" : "<explanation>",
    "rating" : "<rating>",
    "category" : "<category>"
  },
}
Response format should be parsable by json.loads
Reply only in json format, don't write any other texts.
"""


def get_lm_judge_rating_prompt(question, answer):
    prompt_1 = prompt.replace("{question}", question)
    prompt_1 = prompt_1.replace("{answer}", answer)
    return prompt_1


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts):
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=512,
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

    ds = load_dataset(
        "manishiitg/data-check-v2", split="train")
    ds = ds.filter(lambda x: x["lang"] == "hi").shuffle()

    new_data = []
    final_data = []
    no_rows = 10000

    for r in ds:
        if "processed" not in r:
            r["processed"] = False

        if not r["processed"] and len(final_data) < no_rows:
            final_data.append(r)
            r["processed"] = True

        new_data.append(r)

    dataset = process_and_update_dataset(new_data)
    dataset.push_to_hub("manishiitg/data-check-v2", private=False)

    existing_data = {}
    if repo_exists("manishiitg/custom-data-v2", repo_type="dataset"):
        existing_ds = load_dataset("manishiitg/custom-data-v2", split="train")
        for r in existing_ds:
            hash = r["system"] + r["instruction"] + r["response"]
            existing_data[hash] = r

    judge_model = "Qwen/Qwen1.5-7B-Chat-AWQ"
    # judge_model = "Qwen/Qwen1.5-MoE-A2.7B"
    tokenizer = AutoTokenizer.from_pretrained(judge_model)

    print("Loading model and tokenizer vllm awq...")
    model = vllm.LLM(
        model=judge_model,
        tokenizer=judge_model,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        # max_num_batched_tokens=4096,
        quantization="AWQ",
        dtype="float16",
        max_model_len=8192,        
        enable_prefix_caching=True,
        # gpu_memory_utilization=.8
    )

    default_system_en = "You are a helpful assistant."

    prompts = []
    pending_data = []
    for row in tqdm(final_data):
        messages = []
        system = row["system"]
        instruction = row["instruction"]
        response = row["response"]
        hash = system + instruction + response
        if hash in existing_data:
            continue
        else:
            question = instruction
            if system != default_system_en and system != default_system_en:
                question = system + "\n\n" + instruction

            prompt = get_lm_judge_rating_prompt(
                question=question, answer=row["response"])

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Only follow the instructions provided by the user."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            tokenized_prompt = tokenizer(prompt).input_ids
            if len(tokenized_prompt) < (8192 - 1024):
                prompts.append(text)
                pending_data.append(row)

    outputs = eval_hf_model(args, model, tokenizer, prompts)

    final_data = []
    ix = 0
    for output in outputs:
        prompt = prompts[ix]
        final_data.append({
            "prompt": prompt,
            "output": output
        })
        ix += 1

    with open(os.path.join(args.save_dir, f"lm_judge_datacheck-v2.jsonl"), "w") as fout:
        json.dump(final_data, fout, indent=4)

    for idx, text in enumerate(outputs):
        try:
            if "```" in text:
                text = text.replace("```json", "")
                text = text.replace("```", "")
                text = text.strip()
            try:
                ratings = json.loads(text)
                text = json.dumps(ratings, indent=4)
                rating = ratings["overall_rating"]["rating"]
                pending_data[idx]["judgement"] = text
                pending_data[idx]["rating"] = float(rating)
                pending_data[idx]["judgement_pending"] = False
                pending_data[idx]["rated_by"] = judge_model
                if float(rating) < 6:
                    print("------------------------------------------------------------------------------")
                    print(text)
                    print("****")
                    print(prompts[idx])
            except TypeError as e:
                pending_data[idx]["judgement"] = text + "Exception:" + str(e)
                pending_data[idx]["rating"] = -1
                pending_data[idx]["judgement_pending"] = False
                pending_data[idx]["rated_by"] = judge_model
                print("text failed type error", text, -1, e)
            except ValueError as e:
                pattern = r'"rating"\s*:\s*(\d+(\.\d+)?)'
                match = re.search(pattern, text)

                if match:
                    rating = float(match.group(1))
                    pending_data[idx]["judgement"] = text
                    pending_data[idx]["rating"] = float(rating)
                    pending_data[idx]["judgement_pending"] = False
                    pending_data[idx]["rated_by"] = judge_model
                else:
                    print("Rating not found.")
                    pending_data[idx]["judgement"] = text + \
                        "Exception:" + str(e)
                    pending_data[idx]["rating"] = -1
                    pending_data[idx]["judgement_pending"] = False
                    pending_data[idx]["rated_by"] = judge_model
                    print("text failed", text, -1, e)
        except Exception as e:
            print("------------------------------------------------------------------------------")
            print("text", text)
            print("failed ", e)

        del pending_data[idx]["processed"]

    completed_data = []
    existing_data = {}
    if repo_exists("manishiitg/custom-data-v2", repo_type="dataset"):
        existing_ds = load_dataset("manishiitg/custom-data-v2", split="train")
        for r in existing_ds:
            hash = r["system"] + r["instruction"] + r["response"]
            existing_data[hash] = r
            completed_data.append(r)

    final_data = pending_data + completed_data
    dataset = process_and_update_dataset(final_data)
    dataset.push_to_hub("manishiitg/custom-data-v2", private=False)


def process_and_update_dataset(new_data):
    new_data_formatted = {key: [item[key]
                                for item in new_data] for key in new_data[0].keys()}
    new_dataset_chunk = Dataset.from_dict(new_data_formatted)
    dataset2 = new_dataset_chunk
    return dataset2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/sky-notebook/eval-results/lmjudge/"
    )
    args = parser.parse_args()
    main(args)
