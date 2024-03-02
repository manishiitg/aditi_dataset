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


SYSTEM_MESSAGES_ORCA = [
    "",
    "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    "You are an AI assistant that helps people find information. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
    "You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.",
    "Explain how you used the definition to come up with the answer.",
    "You are an AI assistant. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question.",
    "You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer.",
    "User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.",
    "You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.",
    "You are an AI assistant, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines that it provides. You solve the task and show how you used the guidelines to solve the task.",
    "Given a definition of a task and a sample input, break the definition into small parts. Each of those parts will have some instruction. Explain their meaning by showing an example that meets the criteria in the instruction. Use the following format:\n\nPart #: a key part of the definition.\nUsage: Sample response that meets the criteria from the key part. Explain why you think it meets the criteria.",
    "You are an AI assistant that helps people find information.",
]

TOPICS = [
    "Science - physics, chemistry, biology, astronomy, etc.",
    "Mathematics - algebra, geometry, calculus, statistics, etc.",
    "Technology - computers, engineering, AI, robotics, etc.",
    "Business - economics, finance, marketing, management, entrepreneurship",
    "History - ancient, medieval, modern, world history, military history",
    "Geography - countries, capitals, landscapes, maps, oceans, rivers",
    "Literature - poetry, novels, plays, short stories, genres, authors",
    "Philosophy - logic, ethics, political philosophy, existentialism",
    "Psychology - cognition, development, disorders, therapy, social psychology",
    "Sociology - culture, demographics, institutions, social change, inequality",
    "Politics - political systems, ideologies, voting, campaigns, public policy",
    "Law - constitutional law, criminal law, contracts, litigation, civil rights",
    "Medicine - anatomy, diseases, treatments, pharmaceuticals, medical specialties",
    "Religion - Christianity, Islam, Judaism, Buddhism, Hinduism, atheism",
    "Mythology - Greek, Roman, Norse, Egyptian, Native American myths",
    # "Art - art history, painting, sculpture, architecture, music, theater",
    "Sports - individual and team sports, athletes, championships, training",
    "Cooking - recipes, ingredients, techniques, regional and ethnic cuisine",
    "Movies & TV - genre analysis, directors, actors, awards, popular shows",
    "News & Current Events - reporting on latest happenings around the world",
    "Culture - customs, values, gender roles, holidays, language, clothing",
    "Relationships - family, friends, romantic relationships, dating, marriage",
    "Education - teaching methods, curriculum, policy, higher education, vocational",
    "Transportation - cars, planes, trains, ships, public transit, infrastructure",
    "Communication - language acquisition, linguistics, rhetoric, social media",
    "Agriculture - farming techniques, crops, livestock, fisheries, forestry",
    "Housing & Architecture - interior design, urban planning, real estate, remodeling",
    "Nature & Environment - ecology, sustainability, conservation, renewable energy",
    "Travel & Tourism - destinations, lodging, adventure, culture, ecotourism",
    "Music - theory, genres, instruments, bands, composers, music history",
    "Fashion - designers, trends, modeling, retail, cosmetics, accessories",
    "Government - political systems, public administration, foreign policy, voting",
    "Warfare - military history, weapons, strategy, special operations forces",
    "Space - astronomy, spaceflight, exploration, space technology, universe",
    "Weather & Climate - meteorology, forecasting, natural disasters, seasons",
    "Food & Cooking - nutrition, recipes, diets, food science, restaurants",
    # "Pets & Animals - breeds, care, veterinary medicine, wildlife, animal behavior",
    # "Gardening - plants, landscaping, flowers, vegetables, lawn care, tools",
    "Home Improvement - repair, decor, renovation, tools, plumbing, electricity",
    "Personal Finance - budgeting, investing, taxes, insurance, retirement",
    "Exercise & Fitness - techniques, equipment, sports medicine, motivation",
    "Health & Medicine - biology, anatomy, diseases, treatments, wellness",
    # "Mental Health - psychology, disorders, counseling, self-help, mindfulness",
    "Race & Ethnicity - cultures, discrimination, identity, immigration, diversity",
    "Gender & Sexuality - LGBTQ issues, feminism, roles, relationships, equality",
    # "Employment - careers, human resources, resumes, workplace culture, unions",
    "Crime & Justice - laws, law enforcement, courts, prisons, investigations",
    "Social Issues - poverty, homelessness, human rights, community service",
    "Technology - computers, engineering, artificial intelligence, innovations",
    "Entertainment - movies, television, games, comedy, performing arts",
]

SYSTEM_MESSAGES = SYSTEM_MESSAGES_ORCA

PROMPT_2 = """
        You are asked to come up with a set of 50 diverse task instructions. 
        These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

        The instruction should only be related to India with specific context of SUBJECT_AREA

        Here are the requirements:
        1. Try not to repeat the verb for each instruction to maximize diversity.
        2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
        3. The type of instructions should be related to only SUBJECT_AREA
        3.a The type of instruction should not include poem writing
        4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
        5. The instructions should be in English.
        6. The instructions should involve realistic data and should not contain simple placeholders. The instructions should provide substantial content to make the instruction challenging but should ideally not exceed 2 to 3 sentences.
        7. Make sure every instruction captures indian context. 

        List of 50 tasks:

        Example output format in markdown

        *Instruction:* <instruction>
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    topic_instruct_map = {}

    prompts = []
    topics_selected = []
    for idx in tqdm(range(20)):

        topic_number = random.randint(0, len(TOPICS)-1)
        topic_selected = TOPICS[topic_number]

        msg_list = []
        SYSTEM_PROMPT = PROMPT_2 + \
            "\n Question should only be related to india or indian context."

        msg_system = {"role": "system", "content": SYSTEM_PROMPT}
        msg_list.append(msg_system)

        user = f"SUBJECT_AREA: {topic_selected}"

        if topic_selected in topic_instruct_map:
            existing_instruction = topic_instruct_map[topic_selected]
            user += "\n\n" + "Instruction should be different from " + existing_instruction
            
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

    outputs = eval_hf_model(args, model, tokenizer, prompts, .7)

    prompts2 = []
    for idx, text in enumerate(outputs):
        print("======")
        print("prompt", prompts[idx], "text", text)

        # Define the regex pattern to match the instructions
        instruction_pattern = r'\*Instruction:\* (.*?)\n'
        if args.lang == "hi":
            instruction_pattern = r'\*निर्देश:\* (.*?)\n'

        # Find all matches for instructions
        instructions = re.findall(
            instruction_pattern, text, re.DOTALL)

        topic_selected = topics_selected[idx]
        topic_instruct_map[topic_selected] = text

        for inst in instructions:
            print("inst", inst)
            # system_message_number = random.randint(0, len(SYSTEM_MESSAGES)-1)
            # system_message_selected = SYSTEM_MESSAGES[system_message_number]
            # msg_list = []
            # msg_system = {"role": "system", "content": system_message_selected}
            # msg_list.append(msg_system)
            # msg_prompt = {"role": "user", "content": inst}
            # msg_list.append(msg_prompt)
            # text = tokenizer.apply_chat_template(
            #     msg_list,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            # prompts2.append(text)

    # outputs2 = eval_hf_model(args, model, tokenizer, prompts2, .1)
    # for idx, text in enumerate(outputs2):
    #     print("======")

    #     print("text", prompts2[idx])
    #     print("text", text)


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