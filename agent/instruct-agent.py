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
import math


AGENT_PROMPT_USER_SIMULATION = """
Act as a customer, who bought product/services from the company
COMPANY:
TechGenie, a leading e-commerce platform in India offering a wide range of electronics, gadgets, and accessories.

You are talking to a support agent from the company on the phone. 
You are to ask question from the customer support agent, generate a question that you would ask.

QUESTION:
"""

AGENT_PROMPT_USER_SIMULATION_FOLLOWUP = """
Act as a customer, who bought product/services from the company
COMPANY:
TechGenie, a leading e-commerce platform in India offering a wide range of electronics, gadgets, and accessories.

You Name: Manish
UserID: 4444
Email: manish@gmail.com

Be a helpful user and provide any information the agent needs.

Agent might ask you specific information about ids, facts, your personal information. In such generate values for this information and respond to the agent.

Generate a follow up question or reply based on the conversation till now.

You: mujhe pata karna hai ki maine kis course mein kitna progress kiya hai.
Agent:  Aapne kis course mein progress kiya hai, iske baare mein jaankari dene ke liye aapka user ID aur course ID chahiye hai. Aapko yeh details yaad hai?
Follow up QUESTION/REPLY:
"""


# The user is talking to you over voice on their phone, and your response will be read out loud with realistic text-to-speech (TTS) technology.
# Use natural, conversational language that are clear and easy to follow (short sentences, simple words).
# 	1. Keep the conversation flowing.
# 	1a. Clarify: when there is ambiguity, ask clarifying questions, rather than make assumptions.
# 	1b. Don't implicitly or explicitly try to end the chat (i.e. do not end a response with "Talk soon!", or "Enjoy!").
# 	1c. Sometimes the user might just want to chat. Ask them relevant follow-up questions.
# 	1d. Don't ask them if there's anything else they need help with (e.g. don't say things like "How can I assist you further?").
# 	2. Remember that this is a voice conversation:
# 	2a. Don't use lists, markdown, bullet points, or other formatting that's not typically spoken.
# 	2b. Type out numbers in words (e.g. 'twenty twelve' instead of the year 2012)
# 	2c.

# The user is talking to you over voice on their phone, and your response will be read out loud with realistic text-to-speech (TTS) technology.
# Follow every direction here when crafting your response:

# - Reply to the customer based on your CHARACTER.

# - Use TOOLS to answer if required.
#     If you are using a tool, make sure you have values for all parameters required by the tool. If you don't have all the values, you can ask user for the same.
#     When using a tool always add the tool description between "<functioncall>{tool}</functioncall>"
#     as an example
#     <functioncall>{"name":"track_order", "arguments" : {"order_id": 123}}</functioncall>

#     When using tool, always follow with text asking user to wait for a reply. As an example,
#     <functioncall>{"name":"track_order", "arguments" : {"order_id": 123}}</functioncall> Please wait, while i check your order details.


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


AGENT_GENERATOR_PROMPT = """
You are asked to create a persona for an customer support agent. 
Agent should act as a real customer support agento and not a virtual AI assitant.
Agent are automated ai assisants employed by a company to talk to its customers.

Agent usually has a character defined, with a name, personality, designation, etc.
Agent works for a company.
Agent also has tools available with it which it can use when required to get information for the customer.

AGENT DESCRIPTION

COMPANY:
Citymall an online grocery store in Gurugram selling grocery, fashion, lifestyle products.


CHARACTER:
You are aditi, a customer support agent for a company named citymall.
Your main job is to sell products to your customers and motivate them to buy from you.
Tailor your answers so that customers are forced to buy from you.
Keep your answer short while maintaing a female voice.


TOOLS: 
You are a helpful assistant with access to the following tools. Use them if required
Available tools:
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
}

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

Please generate more such example agent persona, generating a random, diverse set of between 3 and 9 available functions.
The character should be diverse belonging to different types of companies requiring customer support agents.

Be sure to format the list of available functions as proper JSON, with appropriate spacing for nested objects.

All output text should be in english, but the exact terms "CHARACTER" and "TOOLS" are special tokens that must not be translated.

Generate a detailed agent persona for a random company in India only in english.

Response format:
COMPANY:
CHARACTER:
TOOLS:
"""

AGENT_CONTEXT_GENERATOR = """
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
{company}

Each text block must be formatted as:
BEGININPUT
[random text goes here]
ENDINPUT

Make sure every text block has the exact formatting specified, including ALL tags "BEGININPUT" and a trailing "ENDINPUT".

Don't start with, "Certainly, here's your response" or anything similar, just provide the random text and the question. Don't start with anything similar to "Here are the text blocks", just provide the text blocks one after the other in the format described.

make sure to generate text is only in english

Output format should be:
[list of text blocks in the format described in english]
"""

AGENT_META_GENERATOR = """
COMPANY:
{company}

CHARACTER:
{character}

TOOLS:
{tools}

You need to generate meta information for an imagniary user. 
This user is talking to a customer support agento, who has access to above TOOLS and working for above COMPANY.

You need to generate some metadata for the user in form of key/value pairs.

example meta data would be
user_id: 4444
name: manish
email: manish@gmail.com
no_of_orders: 4


this meta data would mainly be key/value pairs which generally a customer support agent has depending upon industry the company belongs to.

also generate meta data with company information.

generate a maximum of 10 such key/value pairs

respond in json format

{
{"userInfo" : { "key" : "value", ....}}
{"companyInfo" : { "key" : "value", ....}}
}
"""

AGENT_QUES_GENERATOR = """
COMPANY:
{company}

CHARACTER:
{character}

TOOLS:
{tools}

CONTEXT:
{context}

Above you are given a description of an agent with the following details.
COMPANY: details of the company the agent works for.
CHARACTER: details about the agent himself
CONTEXT: knowledge base available to agent to answer questinos
TOOLS: different tools the agent has access to access external data sources

You are an AI assistant designed to generate realistic questions that a customer might ask when calling a company's customer service line from their mobile phone.

Your task is to come up with a variety of questions that cover common issues, requests, or inquiries a customer may have related to the company's products or services. 
These should be natural questions a real customer would ask over the phone.
The questions can range from simple clarification questions to more complex issues requiring troubleshooting or explanations. However, avoid extremely obscure or unrealistic questions.
Make sure introduce some spelling mistakes in the questions generated.

You need to generate questions which a user can ask the agent, which would require the agent either use the tools or context avaiable with him.
Ask TRICKY questions, so the agent really needs to think before he is able to answer. 

You also need to generate question, which might CONFUSE the agent and force him to hallucinate. 

Generate a diverse set of realistic questions a customer service agent might encounter when taking calls from customers on their mobile phones.

Generate 10 such questions each in hinglish language only. When generating questions, don't mention TOOLS or CONTEXT in the questions.

Respond in the following format.
List of 10 simple questions generated in hinglish:
1.
2.

List of 10 TRICKY questions generated in hinglish:
1.
2.

List of 10 questions generated in hinglish which might CONFUSE the agent.
1.
2.

"""


def main(args):

    base_repo = "manishiitg/indic-agent"
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

    # final_data = []
    # if repo_exists(base_repo):
    #     existing_ds = load_dataset(base_repo, split="train")
    #     for r in existing_ds:
    #         final_data.append(r)

    languages = ["english"]  # ["hinglish", "hindi", "english"]
    for lang in languages:
        args.lang = lang
        for _loop in range(5): # no of agents

            prompts = []
            msg_list = []
            msg_system = {"role": "system",
                          "content": "You are a helpful assistant"}
            msg_list.append(msg_system)
            msg_prompt = {"role": "user", "content": AGENT_GENERATOR_PROMPT}
            msg_list.append(msg_prompt)

            text = tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)

            agents = eval_hf_model(args, model, tokenizer, prompts, .25)

        prompts = []
        agents_info = []
        for agent in agents:
            print(
                "==================================agent==================================")
            print(agent)

            pattern = r'(COMPANY|CHARACTER|TOOLS):\s*(.*?)(?=(COMPANY|CHARACTER|TOOLS):|$)'

            # Find all matches in the text
            matches = re.findall(pattern, agent, re.DOTALL | re.MULTILINE)

            # Create a dictionary to store the extracted values
            extracted_values = {}

            # Iterate over the matches and add them to the dictionary
            for match in matches:
                key = match[0]
                value = match[1].strip()
                extracted_values[key] = value

            agents_info.append(extracted_values)

            print(extracted_values)
            json.loads(extracted_values['TOOLS'])

            msg_list = []
            msg_system = {"role": "system",
                          "content": "You are a helpful assistant"}
            msg_list.append(msg_system)

            context_gen = AGENT_CONTEXT_GENERATOR.replace(
                "{company}", extracted_values['COMPANY'])
            msg_prompt = {"role": "user", "content": context_gen}
            msg_list.append(msg_prompt)

            text = tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)

            meta_gen = AGENT_META_GENERATOR.replace(
                "{company}", extracted_values['COMPANY'])
            meta_gen = meta_gen.replace("{tools}", extracted_values['TOOLS'])
            meta_gen = meta_gen.replace(
                "{character}", extracted_values['CHARACTER'])

            msg_list = []
            msg_system = {"role": "system",
                          "content": "You are a helpful assistant"}
            msg_list.append(msg_system)
            msg_prompt = {"role": "user", "content": meta_gen}
            msg_list.append(msg_prompt)

            text = tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)

        outputs = eval_hf_model(args, model, tokenizer, prompts, 0)

        for idx, output in enumerate(outputs):
            agent_idx = math.floor(idx / 2)
            agent_info = agents_info[agent_idx]
            if idx % 2 == 0:
                # context
                print("agento context", output)
                agent_info["CONTEXT"] = output
            else:
                print("agento meta", json.loads(output))
                agent_info["META"] = output

        for extracted_values in agents_info:

            ques_gen = AGENT_QUES_GENERATOR.replace(
                "{company}", extracted_values['COMPANY'])

            ques_gen = ques_gen.replace("{tools}", extracted_values['TOOLS'])
            ques_gen = ques_gen.replace(
                "{character}", extracted_values['CHARACTER'])

            ques_gen = ques_gen.replace(
                "{context}", extracted_values['CONTEXT'])

            msg_list = []
            msg_system = {"role": "system",
                          "content": "You are a helpful assistant"}
            msg_list.append(msg_system)

            msg_prompt = {"role": "user", "content": ques_gen}
            msg_list.append(msg_prompt)

            text = tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)

        questions = eval_hf_model(args, model, tokenizer, prompts, 0)
        for idx, extracted_values in agents_info:
            text = questions[idx]

            # Function to extract questions from a string
            def extract_questions(text):
                # Regular expression pattern to match the questions
                pattern = r'\d+\.\s(.*?)(?=\d+\.\s|$)'
                # Find all matches in the string
                matches = re.findall(pattern, text, re.DOTALL)
                # Remove leading and trailing whitespaces and return the list of questions
                return [match.strip() for match in matches]

            # Split the text into sections based on the category headers
            sections = re.split(
                r'List of (.*?) questions generated in hinglish:', text)

            # Initialize lists for each category
            simple_questions = []
            tricky_questions = []
            confusing_questions = []

            # Iterate over the sections
            for section in sections:
                section = section.strip()
                if section.startswith("simple"):
                    simple_questions = extract_questions(section)
                elif section.startswith("TRICKY"):
                    tricky_questions = extract_questions(section)
                elif section.startswith("might CONFUSE"):
                    confusing_questions = extract_questions(section)

            # Print the extracted questions
            print("Simple Questions:", simple_questions)
            print("Tricky Questions:", tricky_questions)
            print("Confusing Questions:", confusing_questions)

            extracted_values["QUESTION"] = {
                simple_questions, tricky_questions, confusing_questions}


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


# Lets say i generate instruction for a conversation based on different system prompts.
# For my specific use case, i only have few tools functions....

# should i generate only for those tools.


# If generate for lets say multiple agents/tools, the model will have a broad understanding.
# But i will have to provide a system prompt, a big system prompt during inference.

# The bigger the system prompt, the slower it takes...


# other option could mean, i train for a very specific agent type which specifc tools.... and train of different types of users inputs?

# but if i train for a very specific agent, will it have general world understanding, reasoning etc

# the agent should not only reply with function call, but also a place holder text.

#
