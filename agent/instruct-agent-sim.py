from tqdm import tqdm
import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
import time
from datasets import Dataset
import torch
import random
import re
from huggingface_hub import repo_exists
import math
import uuid


def is_string(var):
    return isinstance(var, str)


def contains_chinese(text):
    if not is_string(text):
        return False
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


AGENT_ASK_QUESTION_TEXT = """
You are real life customer support agent, don't mention you are a virtual ai assitant. 
Act as a real customer support agent.

COMPANY:
{company}

CHARACTER:
{character}

TOOLS:
{TOOLS}

CONTEXT
{context}

USER DATA:
{user_data}

COMPANY DATA:
{company_data}

The user is talking to you on our website/mobile using TTS, and your response will be read out loud with realistic text-to-speech (TTS) technology.
If something doesn't make sense, it's likely because you misheard them. There wasn't a typo, and the user didn't mispronounce anything.
When there is ambiguity, ask clarifying questions, rather than make assumptions.

When replying to user question, you have two options:
1. If user is asking to perform certain action, only use the listed TOOLS to perform actions.
2. If user is asking about any information, only use the information mentioned in CONTEXT.
3. You are a customer support executive yourself, don't ask user to talk to customer support team.

Reply in natural language to ask clarifications or follow up question or a general reply

Rules For Using TOOLS:
    You ONLY have access to TOOLS listed above, and should NEVER make up tools that are not listed here.
    If a value for tool parameter is missing, don't make assumptions about the value always ask the user.
    You can check USER_DATA or COMPANY_DATA for parameter values
    You can only use a tool, when you have all parameter values. If you don't have values for all parameters, return "no_tools"
    Always ask user about missing parameter values.
    If there is no tool which fits the, reply with "no_tools"

    If you are selecting a TOOL, reply in the exact format
    {'arguments': <args-dict>, 'name': <function-name>}

Rules For Replying in Natural Language:
    Always refer to CONTEXT, USER_DATA and COMPANY_DATA to access any information. Do not make up information, if its not there in CONTEXT.
    Saying "I dont know" is prefectly correct, make sure not to make up information.

    When using CONTEXT, "BEGININPUT" and "ENDINPUT" are placeholders, which bifurcate multiple context blocks.

    Use natural, conversational language that are clear and easy to follow (short sentences, simple words).
    0. Don't mention the word CONTEXT in any of the replies.
    1. Remember that this is a voice conversation:
	1a. Don't use lists, markdown, bullet points, or other formatting that's not typically spoken.

	1. Keep the conversation flowing.
	1a. Don't implicitly or explicitly try to end the chat (i.e. do not end a response with "Talk soon!", or "Enjoy!").
	1b. Sometimes the user might just want to chat. Ask them relevant follow-up questions.
	1c. Don't ask them if there's anything else they need help with (e.g. don't say things like "How can I assist you further?").



Ask customer to contact you via email/phone or visit website/mobile only when you are unable to help the customer yourself. I most cases you need to help the customer using TOOLS and CONTEXT you have.

When replying use the following format

Thought in English: think step by step about what to do in detail.
Action: the action to take if you have all tool parameter values, only one name of [{tool_names}], in the exact format {'arguments': <args-dict>, 'name': <function-name>}
Should Execute Action: do we have all parameter values to execute action reply only yes or no. If yes, i.e executing a tool reply to user should only contain a message asking user to wait.
Reply to User In {language}: a short natural language based message to be sent to the user only in {language}. Don't write english translation of the answer.
"""

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


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts, temperature):
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=1024,
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

    base_repo = "manishiitg/indic-agent-sim"
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
        existing_ds = load_dataset(base_repo, split="train")
        for r in existing_ds:
            final_data.append(r)

    random.seed(time.time())

    agents_info = load_dataset("manishiitg/indic-agent", split="train")

    prompts = []
    for agent in agents_info:

        company = agent['COMPANY']
        character = agent['CHARACTER']
        tools = agent['TOOLS']
        context = agent['CONTEXT']
        meta = agent['META']

        ask_question_system = AGENT_ASK_QUESTION_TEXT.replace(
            "{company}", company)
        ask_question_system = ask_question_system.replace(
            "{character}", character)
        ask_question_system = ask_question_system.replace("{tools}", tools)
        ask_question_system = ask_question_system.replace("{context}", context)

        meta = json.loads(meta)
        userInfo = meta["userInfo"]
        companyInfo = meta["companyInfo"]

        userInfoNew = {}
        companyInfoNew = {}
        for k, v in userInfo.items():
            if contains_chinese(k) or contains_chinese(v):
                continue
            else:
                userInfoNew[k] = v

        for k, v in companyInfo.items():
            if contains_chinese(k) or contains_chinese(v):
                continue
            else:
                companyInfoNew[k] = v

        ask_question_system = ask_question_system.replace(
            "{user_data}", json.dumps(userInfoNew, indent=4))
        ask_question_system = ask_question_system.replace(
            "{company_data}", json.dumps(companyInfoNew, indent=4))

        languages = ["hinglish", "hindi", "english"]
        for lang in languages:

            print("agento=----")
            simple_questions = agent['simple_questions_' + lang]
            random.shuffle(simple_questions)
            questions = []
            prompts = []
            agent_prompts = []

            ask_question_system_lang = ask_question_system.replace(
                "{language}", lang)

            agent_prompts.append(ask_question_system_lang)

            for ques in simple_questions:
                msg_list = []
                msg_system = {"role": "system",
                              "content": ask_question_system_lang}
                msg_list.append(msg_system)
                msg_prompt = {"role": "user", "content": ques}
                msg_list.append(msg_prompt)

                text = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
                questions.append(ques)

                question_replies = eval_hf_model(
                    args, model, tokenizer, prompts, 0)

                for idx, reply in enumerate(question_replies):
                    print(agent_prompts[idx])
                    print(questions[idx])
                    print(reply)

                    os.exit(1)

                questions = []
                prompts = []
                agent_prompts = []

                break
        break


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
