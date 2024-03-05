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
from huggingface_hub import repo_exists


SYSTEM_MESSAGES_ORCA = [
    "",
    "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    "You are an AI assistant that helps people find information. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
    "You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer.",
    "User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.",
    "You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.",
    "You are an AI assistant that helps people find information.",
]

TOPICS = [
    "Cricket - players, matches, history, etc.",
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
    # "Social Issues - poverty, homelessness, human rights, community service",
    "Technology - computers, engineering, artificial intelligence, innovations",
    "Entertainment - movies, television, games, comedy, performing arts",
]

TOPICS_INDIA = [
    "Bollywood movies and the influence of Indian cinema",
    "Yoga and its origins in ancient Indian traditions",
    "The rise of Indian startups and the tech industry",
    "Traditional Indian cuisine and the popularity of regional dishes",
    "The impact of social media on Indian youth culture",
    "Indian classical music and its rich heritage",
    "The growing popularity of Indian fashion and textiles",
    "The role of religion in Indian society",
    "The challenges of urbanization and rapid development in Indian cities",
    "The influence of Indian literature and authors",
    "The rise of Indian influencers and content creators",
    "The impact of the Indian diaspora on global culture",
    "The popularity of Indian sports like cricket and kabaddi",
    "The growth of the Indian film industry beyond Bollywood",
    "The increasing focus on sustainable living and eco-friendly practices in India",
    "The impact of the COVID-19 pandemic on Indian society",
    "The rise of Indian entrepreneurs and business leaders",
    "The popularity of traditional Indian arts and crafts",
    "The challenges of gender inequality and women's empowerment in India",
    "The impact of Indian philosophy and spiritual traditions",
    "The growth of the Indian education sector and educational reforms",
    "The popularity of Indian festivals and celebrations",
    "The impact of the Indian diaspora on global cuisines",
    "The rise of Indian stand-up comedy and comedians",
    "The challenges of poverty and income inequality in India",
    "The popularity of Indian music genres like Bollywood and Punjabi",
    "The impact of Indian mythology and epics on popular culture",
    "The growth of the Indian healthcare sector and medical tourism",
    "The popularity of Indian weddings and their traditions",
    "The impact of Indian politics and the role of political parties",
    "The growth of the Indian tourism industry and popular destinations",
    "The popularity of Indian festivals and their cultural significance",
    "The impact of Indian architecture and heritage sites",
    "The rise of Indian-origin tech entrepreneurs and their global impact",
    "The popularity of Indian dance forms like Bharatanatyam and Kathak",
    "The impact of Indian spirituality and wellness practices",
    "The growth of the Indian gaming industry and esports",
    "The popularity of Indian street food and its regional variations",
    "The impact of Indian art and artists on the global art scene",
    "The rise of Indian influencers in the beauty and fashion industry",
    "The popularity of Indian festivals and their cultural significance",
    "The impact of Indian music and its influence on global music",
    "The growth of the Indian automotive industry and electric vehicles",
    "The popularity of Indian spices and their health benefits",
    "The impact of Indian literature and its influence on global literature",
    "The rise of Indian-origin actors and their global success",
    "The popularity of Indian handicrafts and their cultural significance",
    "The impact of Indian cuisine on global food trends",
    "The growth of the Indian media and entertainment industry",
    "The popularity of Indian festivals and their cultural significance",

    "The rise of online food delivery services in Indian cities",
    "The popularity of traditional Indian board games like Chaupar and Pachisi",
    "The impact of the Indian Premier League (IPL) on cricket fandom",
    "The growth of the Indian beauty and wellness industry",
    "The popularity of Indian classical dance forms like Kathak and Bharatanatyam",
    "The challenges of water scarcity and conservation in India",
    "The rise of Indian-origin entrepreneurs in the tech industry",
    "The popularity of Indian festivals like Diwali and Holi",
    "The impact of Bollywood on Indian fashion and style trends",
    "The growth of the Indian e-commerce and online shopping industry",
    "The popularity of Indian folk music and its regional variations",
    "The impact of the Indian caste system on social mobility",
    "The rise of Indian-origin chefs and their influence on global cuisine",
    "The popularity of Indian martial arts like Kalaripayattu and Thang-Ta",
    "The impact of the Indian independence movement on national identity",
    "The growth of the Indian renewable energy sector",
    "The popularity of Indian street food and its regional variations",
    "The impact of Indian mythology and folklore on popular culture",
    "The rise of Indian stand-up comedy and its popularity among youth",
    "The popularity of Indian traditional medicine like Ayurveda and Unani",
    "The impact of the Indian education system on social and economic mobility",
    "The growth of the Indian textile industry and the popularity of handloom products",
    "The popularity of Indian music genres like Carnatic and Hindustani classical",
    "The impact of the Indian diaspora on global culture and cuisine",
    "The rise of Indian-origin actors and their success in Hollywood",
    "The popularity of Indian festivals and their cultural significance",
    "The impact of the Indian IT industry on the global tech landscape",
    "The growth of the Indian sports industry beyond cricket",
    "The popularity of Indian spices and their health benefits",
    "The impact of the Indian caste system on social and economic inequality",
    "The rise of Indian-origin authors and their influence on global literature",
    "The popularity of Indian traditional crafts like Zardozi and Phulkari",
    "The impact of the Indian independence movement on gender equality",
    "The growth of the Indian tourism industry and popular destinations",
    "The popularity of Indian vegetarian cuisine and its regional variations",
    "The impact of the Indian space program on scientific advancement",
    "The rise of Indian-origin entrepreneurs in the food industry",
    "The popularity of Indian classical music and its regional variations",
    "The impact of the Indian film industry on popular culture and social norms",
    "The growth of the Indian automotive industry and the rise of electric vehicles",
    "The popularity of Indian festivals and their cultural significance",
    "The impact of the Indian caste system on social mobility and economic opportunity",
    "The rise of Indian-origin entrepreneurs in the fashion industry",
    "The popularity of Indian traditional arts like Rangoli and Mehndi",
    "The impact of the Indian independence movement on religious harmony",
    "The growth of the Indian sports industry and the rise of new sports leagues",
    "The popularity of Indian street food and its regional variations",
    "The impact of the Indian education system on the growth of the IT industry",
    "The rise of Indian-origin entrepreneurs in the healthcare industry",
    "The popularity of Indian festivals and their cultural significance",
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

1. <instruction>
2. <instruction>
"""

PROMPT_4 = """
You are asked to come up with a set of 50 diverse task instructions. 
These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

The instruction should only be related to India with specific context of SUBJECT_AREA

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be related to only SUBJECT_AREA
    3.a The type of instruction should not include poem writing
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, knowledge based questions, classification, editing, etc.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in hinglish.
6. The instructions should involve realistic data and should not contain simple placeholders. The instructions should provide substantial content to make the instruction challenging but should ideally not exceed 2 to 3 sentences.
7. Make sure every instruction captures indian context and is in in hinglish language only. 

List of 50 tasks:

1. <instruction_in_hinglish>
2. <instruction_in_hinglish>
"""

PROMPT_5 = """
You are asked to come up with a set of 20 diverse task instructions. 
These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

The instruction should only be related to "The influence of TikTok on social media trends"

Here are the requirements:
1. Instructions generated should only be in hinglish.
2. Try not to repeat the verb for each instruction to maximize diversity.
3. The type of instructions should be diverse. 
    a. The list should include diverse types of tasks like open-ended generation, classification, editing, writing, conversation, trivia, etc.
4. The type of instructions should be related to only "The influence of TikTok on social media trends"
5 . A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted
7. The instructions should involve realistic data and should not contain simple placeholders. The instructions should provide substantial content to make the instruction challenging but should ideally not exceed 2 to 3 sentences.

List of 20 tasks:

1. <instruction_in_hinglish>
2. <instruction_in_hinglish>

Generate tasks only in hinglish not hindi. Translate tasks generated to hinglish if required.
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

    base_repo = "manishiitg/indic-synthetic-instruct"
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
    if repo_exists(base_repo):
        existing_ds = load_dataset(base_repo, split="train")
        for r in existing_ds:
            final_data.append(r)

    global TOPICS
    languages = ["hinglish"]  # ["hinglish", "hindi", "english"]
    for lang in languages:
        args.lang = lang
        topic_instruct_map = {}
        for loop in range(10):

            prompts = []
            topics_selected = []
            random.shuffle(TOPICS)
            if args.generate_topics:
                prompt = """Give me a list of 50 completely random topics, related to india, indian culture, indian socity, latest trends in india and what people talk about in indian"""

                outputs = eval_hf_model(args, model, tokenizer, [prompt], .7)
                output = outputs[0]
                print("topics generated", output)
                topics = output.split("\n")
                TOPICS = []
                for t in topics:
                    idx = t.index(".")
                    if idx != -1:
                        t = t[idx + 1:]
                    TOPICS.append(t)
                    print("topic", t)

            for topic_selected in TOPICS:

                # topic_number = random.randint(0, len(TOPICS)-1)
                # topic_selected = TOPICS[topic_number]

                msg_list = []
                SYSTEM_PROMPT = PROMPT_2

                if args.lang == "hindi":
                    SYSTEM_PROMPT = PROMPT_5

                if args.lang == "hinglish":
                    SYSTEM_PROMPT = PROMPT_4

                user = f"SUBJECT_AREA: {topic_selected}"

                if topic_selected in topic_instruct_map:
                    existing_instruction = topic_instruct_map[topic_selected]
                    user += "\n\n" + "Generated Instructions should be different from " + existing_instruction

                if args.model_name_or_path == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                    msg_prompt = {"role": "user",
                                  "content": SYSTEM_PROMPT + "\n\n" + user}
                    msg_list.append(msg_prompt)
                else:
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

                instructions = []
                matches = text.split("\n")
                for match in matches:
                    if "." in match:
                        ix = match.index(".")
                        match = match[ix+1:]
                    else:
                        print("skipping instruction", match)
                        continue
                    match = match.strip()
                    if match.startswith('"'):
                        match = match[0:]
                    if match.endswith('"'):
                        match = match[:-1]
                    instructions.append(match.strip())

                topic_selected = topics_selected[idx]
                topic_instruct_map[topic_selected] = text

                for inst in instructions:
                    print("inst", inst)
                    system_message_number = random.randint(
                        0, len(SYSTEM_MESSAGES)-1)
                    system_message_selected = SYSTEM_MESSAGES[system_message_number]
                    if args.lang == "hindi":
                        system_message_selected += "\n\nAnswer in hindi only"
                    if args.lang == "hinglish":
                        system_message_selected += "\n\nAnswer in hinglish only"
                    if args.model_name_or_path == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                        msg_list = []
                        msg_prompt = {
                            "role": "user", "content": system_message_selected + "\n\n" + inst}
                        msg_list.append(msg_prompt)
                    else:
                        msg_list = []
                        msg_system = {"role": "system",
                                      "content": system_message_selected}
                        msg_list.append(msg_system)
                        msg_prompt = {"role": "user", "content": inst}
                        msg_list.append(msg_prompt)

                    text = tokenizer.apply_chat_template(
                        msg_list,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts2.append(text)
                    topics_selected2.append(topic_selected)
                    sys_prompt_selected.append(system_message_selected)
                    question2.append(inst)

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
