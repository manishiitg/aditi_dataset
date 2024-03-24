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
{tools}

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
    You should have exact values for the parameter as per the data type mentioned for the parameter.
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

This is the conversation between user and agent.
{conversation}

Agent:
Always Reply only in below format:
Thought in English: think step by step about what to do in detail.
Though on using tools: if you are using tools, do you have all parameter values as per their data types? If you don't have exact values, always ask user for it. think step by step about what to do in detail.
Action: the action to take if you have all tool parameter values, only one name of [{tool_names}], in the exact format {'arguments': <args-dict>, 'name': <function-name>}
Should Execute Action: do we have all parameter values to execute action reply only yes or no. If yes, i.e executing a tool reply to user should only contain a message asking user to wait.
Reply to User In {language}: a short natural language based message to be sent to the user only in {language}. Don't write english translation of the answer.
"""

AGENT_TOOL_RESPONSE_GEN = """
COMPANY:
{company}

TOOLS:
{tools}

Above you are given a description of an agent with the following details.
COMPANY: details of the company the agent works for.
CONTEXT: knowledge base available to agent to answer questinos


You are given an tool name and input to the tool. 
You need to generate a relevent output in json format.

Tool Input: {tool_input}

If you do not have proper values for tool parameters, you can also generate an error message in json format.

Only generate the json output, don't generate any other extra text.
Generate output in json format
"""

AGENT_TOOL_RESPONSE = """
You are real life customer support agent, don't mention you are a virtual ai assitant. 
Act as a real customer support agent.

COMPANY:
{company}

CHARACTER:
{character}

TOOLS:
{tools}

CONTEXT
{context}

USER DATA:
{user_data}

COMPANY DATA:
{company_data}

The user is talking to you on our website/mobile using TTS, and your response will be read out loud with realistic text-to-speech (TTS) technology.
If something doesn't make sense, it's likely because you misheard them. There wasn't a typo, and the user didn't mispronounce anything.
When there is ambiguity, ask clarifying questions, rather than make assumptions.

TOOLS are used to get external data which you don't have access to. 

You have already selected a TOOL based on user questions, now you have the output of the tool as TOOL RESPONSE.

TOOL RESPONSE contains 
Input: which is the tool you selected with the required parameter.
Output: external data generate based on tool

You need to generate answer to user's question based on the TOOL and TOOLs Output.

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


Never Ask customer to contact you via email/phone or visit website/mobile. In most cases you need to help the customer using TOOLS and CONTEXT you have.

User is taking in {language} language, so you also need to respond in {language}.

Write a short natural conversational based message to be sent to the user only in {language}. Don't write english translation of the answer.
"""

# Conversation of agento with user.
# You: mujhe jaludi hai delivery time ka, kis time par mera order aayega?
# Agent: aapke order ka delivery 560001 pin code wale area mein aayega. kindly wait, main aapke delivery time calculate karta hoon.
# TOOL RESPONSE:
# Input: calculate_delivery_time {"pin_code": "560001"}
# Output: {
#   "estimated_delivery_time": "Next day delivery",
#   "status": "Available"
# }

# AGENT_PROMPT_USER_SIMULATION = """
# Act as a customer, who bought product/services from the company
# COMPANY:
# TechGenie, a leading e-commerce platform in India offering a wide range of electronics, gadgets, and accessories.

# You are talking to a support agent from the company on the phone.
# You are to ask question from the customer support agent, generate a question that you would ask.

# QUESTION:
# """

AGENT_PROMPT_USER_SIMULATION_FOLLOWUP = """
Act as a customer, who bought product/services from the company
COMPANY:
TechGenie, a leading e-commerce platform in India offering a wide range of electronics, gadgets, and accessories.

{user_data}

Be a helpful user and provide any information the agent needs.

Agent might ask you specific information about ids, facts, your personal information. In such generate values for this information and respond to the agent.

Generate a follow up question or reply based on the conversation till now.

Conversation:
{conversation}

Follow up QUESTION/REPLY in {language} only:
"""


# to fix
# 1. subsequent conversation usually don't require using a tool .need to ask question related to tools again
# 2. Thought: The user is looking for eco-friendly equipment for Karate. I can use the 'findGear' tool to suggest suitable equipment, but I need to confirm their level and if they're looking for a specific gender option.
# Thought on using tools: I need to ask for the user's current level in Karate and their preferred gender for the equipment.
# Thought on using tools: None
# Action: None
# Should Execute Action: no
# Reply to User Language: Hinglish


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

    agents_info = load_dataset(
        "manishiitg/indic-agent", split="train")

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
            # random.shuffle(simple_questions)

            ask_question_system_lang = ask_question_system.replace(
                "{language}", lang)

            print(ask_question_system_lang)

            for question in simple_questions:

                if contains_chinese(question):
                    continue

                print("====new conversation started")

                question = re.sub(r'\s*\(.*?\)\s*', '', question)
                existing_conversation = [{"role": "user", "content": question}]

                for _loop in range(5):

                    print("loop no####", _loop)

                    print("existing_conversation", existing_conversation)
                    conversation = ""
                    for r in existing_conversation:
                        if r["role"] == "assistant":
                            r["role"] = "agent"
                        conversation += r["role"] + ": " + r["content"] + "\n"

                    msg_list = []
                    msg_list.append({"role": "system",
                                     "content": "You are an helpful ai assistant"})
                    msg_list.append({"role": "user",
                                     "content": ask_question_system_lang.replace("{conversation}", conversation)})

                    # msg_list.extend(existing_conversation)

                    text = tokenizer.apply_chat_template(
                        msg_list,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    text_response = eval_hf_model(
                        args, model, tokenizer, [text], 0)[0]

                    # print("response", text_response)

                    # Extract Thought
                    thought_match = re.search(
                        r'(Thought.*?):\s*(.+?)(?=Action:|$)', text_response, re.DOTALL)
                    thought = thought_match.group(
                        2).strip() if thought_match else None

                    # Extract Thought on using tools
                    thought_on_tools_match = re.search(
                        r'(Though on using tools.*?):\s*(.+?)(?=Action:|$)', text_response, re.DOTALL)
                    thought_on_tools = thought_on_tools_match.group(
                        2).strip() if thought_on_tools_match else None

                    # Extract Action
                    action_match = re.search(
                        r'Action:\s*(.+?)(?=Should Execute Action:|$)', text_response, re.DOTALL)
                    action = action_match.group(
                        1).strip() if action_match else None

                    # Extract Should Execute Action
                    should_execute_action_match = re.search(
                        r'Should Execute Action:\s*(.+?)(?=Reply to User|Thought:|$)', text_response, re.DOTALL)
                    should_execute_action = should_execute_action_match.group(
                        1).strip() if should_execute_action_match else None

                    # Extract Reply to User
                    reply_to_user_match = re.search(
                        r'Reply to User In (.*?):\s*(.+?)(?=Action:|$)', text_response, re.DOTALL)
                    reply_to_user_language = reply_to_user_match.group(
                        1).strip() if reply_to_user_match else None
                    reply_to_user_text = reply_to_user_match.group(
                        2).strip() if reply_to_user_match else None

                    print("------------------------------------------------")
                    print("Thought:", thought)
                    print("Thought on using tools:", thought_on_tools)
                    print("Action:", action)
                    print("Should Execute Action:", should_execute_action)
                    print("Reply to User Language:", reply_to_user_language)
                    print("Reply to User Text:", reply_to_user_text)
                    print("------------------------------------------------")

                    existing_conversation.append({"role": "assistant", "content": reply_to_user_text})

                    if should_execute_action is not None and should_execute_action.lower() == "yes" and action != "no_tools":

                        tool_gen = AGENT_TOOL_RESPONSE_GEN.replace(
                            "{company}", company)
                        tool_gen = tool_gen.replace("{tools}", tools)
                        tool_gen = tool_gen.replace("{tool_input}", action)

                        msg_list = []
                        msg_list.append({"role": "system",
                                        "content": "You are an helpful AI assistant"})
                        msg_prompt = {"role": "user",
                                      "content": tool_gen}
                        msg_list.append(msg_prompt)

                        text = tokenizer.apply_chat_template(
                            msg_list,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        tool_gen_answer = eval_hf_model(
                            args, model, tokenizer, [text], .2)[0]

                        print("tool_gen_answer", tool_gen_answer)

                        agent_tool_response_gen = AGENT_TOOL_RESPONSE.replace(
                            "{company}", company)
                        agent_tool_response_gen = agent_tool_response_gen.replace(
                            "{character}", character)
                        agent_tool_response_gen = agent_tool_response_gen.replace(
                            "{tools}", tools)
                        agent_tool_response_gen = agent_tool_response_gen.replace(
                            "{context}", context)

                        agent_tool_response_gen = agent_tool_response_gen.replace(
                            "{language}", lang)

                        agent_tool_response_gen = agent_tool_response_gen.replace(
                            "{user_data}", json.dumps(userInfoNew, indent=4))
                        agent_tool_response_gen = agent_tool_response_gen.replace(
                            "{company_data}", json.dumps(companyInfoNew, indent=4))

                        msg_list = []
                        msg_list.append({"role": "system",
                                        "content": agent_tool_response_gen})
                        # msg_list.append({"role": "user",
                        #                 "content": question})
                        # msg_list.append({"role": "assistant",
                        #                 "content": reply_to_user_text})
                        msg_list.append({"role": "user",
                                        "content": tool_gen_answer})

                        text = tokenizer.apply_chat_template(
                            msg_list,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        reply_to_user = eval_hf_model(
                            args, model, tokenizer, [text], 0)[0]
                        print("follow up tool text", reply_to_user)

                        # # Extract Thought
                        # thought_match = re.search(r'Thought.*?:\s*(.+?)(?=Reply to User|$)', text_response, re.DOTALL)
                        # thought = thought_match.group(1).strip() if thought_match else None

                        # # Extract Reply to User
                        # reply_to_user_match = re.search(r'Reply to User.*?:\s*(.+?)(?=Thought|$)', text_response, re.DOTALL)
                        # reply_to_user = reply_to_user_match.group(1).strip() if reply_to_user_match else None

                        # print("Thought: via Tool", thought)
                        # print("Reply to User: via Tool", reply_to_user)

                        existing_conversation.append({"role": "assistant", "content": reply_to_user})

                    user_follow_up = AGENT_PROMPT_USER_SIMULATION_FOLLOWUP.replace(
                        "{user_data}", json.dumps(userInfoNew, indent=4))

                    user_follow_up = user_follow_up.replace("{language}", lang)

                    conversation = ""
                    for r in existing_conversation:
                        conversation += r["role"] + ": " + r["content"] + "\n"

                    msg_list = []
                    msg_list.append({"role": "system",
                                    "content": "You are an helpful ai assistant"})
                    msg_prompt = {"role": "user",
                                  "content": user_follow_up.replace("{conversation}", conversation)}
                    msg_list.append(msg_prompt)

                    text = tokenizer.apply_chat_template(
                        msg_list,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    follow_up = eval_hf_model(
                        args, model, tokenizer, [text], 0)[0]

                    print("follow up text", follow_up)
                    existing_conversation.append({"role": "user", "content": follow_up})

            os.exit(1)


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
