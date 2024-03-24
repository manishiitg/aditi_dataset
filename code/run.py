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


PROMPT_1 = """
I would like you to help me create a set of coding and/or scripting tasks in hindi language.

Here are a few example tasks:
Example 1. एक एसिंक पायथन फास्टएपीआई सर्वर लिखें। वेबसर्वर को पोर्ट, लिसनिंग आईपी और फ़ाइलों की सेवा करने और फ़ाइलों को सहेजने के लिए एक स्थिर निर्देशिका के लिए कमांड लाइन तर्कों का समर्थन करना चाहिए। स्थिर फ़ाइलों के अलावा, वेबसर्वर को एक अपलोड एंडपॉइंट का समर्थन करना चाहिए, जो फ़ाइल नाम के लिए सामग्री के sha256 हेक्सडाइजेस्ट का उपयोग करके फ़ाइलों को निर्दिष्ट निर्देशिका में सहेजता है। यदि फ़ाइल पहले से मौजूद है, तो अपलोड को छोड़ दिया जाना चाहिए। इसके अलावा, सभी फ़ाइलों को सूचीबद्ध करने और आईडी द्वारा फ़ाइल प्राप्त करने के लिए एक समापन बिंदु होना चाहिए।

Example 2: "Person" के लिए एक पायथन मॉडल बनाएं जो सत्यापनकर्ताओं के साथ-साथ कार्यस्थल के संदर्भ में अच्छा काम करेगा, और व्यक्ति से संबंधित दूरस्थ जानकारी जैसे सोशल मीडिया प्रोफाइल/सामग्री लाने का एक तरीका होगा। दूरस्थ फ़ाइल और क्रॉप/आकार बदलने के समर्थन के साथ एक प्रोफ़ाइल छवि शामिल करना सुनिश्चित करें।

Example 3: जावास्क्रिप्ट में बी*ट्री एल्गोरिदम लागू करें।

Example 4: GoLang में जिकस्ट्रा के एल्गोरिदम का समाधान लिखें।

Example 5. बिना किसी ग्राफिकल लाइब्रेरी के, पायथन में एक एएससीआईआई स्नेक गेम बनाएं।

Example 6. पायथन का उपयोग करके टेक्स्ट-आधारित सॉलिटेयर कार्यान्वयन बनाएं। यहां कुछ अतिरिक्त आवश्यकताएं हैं:
 - खेल स्थिति को mariadb में सहेजा जाना चाहिए।
 - गेम को प्रत्येक चाल के बाद रिकॉर्ड के रूप में एक स्क्रीनशॉट लेना चाहिए।
 - गेम को केवल कीबोर्ड इनपुट स्वीकार करना चाहिए, माउस नहीं।

Example 7. निम्नलिखित के साथ एक पायथन स्क्रिप्ट बनाएं:
 एक। एक शब्द को इनपुट के रूप में स्वीकार करता है।
 बी। यदि उपयोगकर्ता एक से अधिक शब्द इनपुट करता है, या शब्द 10 अक्षरों से अधिक है, तो एक त्रुटि संदेश प्रिंट करें और रोकें।
 सी। इनपुट शब्द का एक बड़ा एएससीआई आर्ट संस्करण तैयार करें।

The tasks must be something a coding language model can complete without access to additional resources.

The tasks must be something that could be completed with 2000 words and symbols or less.

Task should be related to language "{programming_language}"

The tasks should be in Hindi language.

None of the tasks should be about reading from or writing to csvs.

Give me a numbered list of 10 new coding tasks in hindi.

Tasks should all details required to write the code, including programing language to use or any other data required to completed the task. 

Don't mention the word "hindi" in tasks generated.

Response format:
TSK 1. [task 1 in hindi]
TSK 2. [task 2 in hindi]
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
"""

PROMPT_2 = """
I would like you to help me create a set of coding and/or scripting tasks in hinglish language.

Here are a few example tasks:
Example 1. Ek async python FastAPI server likho. Yeh webserver port, listening IP, aur files ko serve aur save karne ke liye ek static directory ke liye command line arguments ka support karna chahiye. Static files ke alava, webserver ko ek upload endpoint ka support karna chahiye, jo files ko specified directory mein save karta hai, file name ke liye content ka sha256 hexdigest ka use karke. Agar file pahle se exist karta hai, to upload ko discard kar diya jana chahiye. Iske alava, sabhi files ko list karne aur ID dwara file prapt karne ke liye ek endpoint hona chahiye.

Example 2: "Person" ke liye ek python model banao jo workplace context mein achhe se kaam kare, saath hi validators, aur person se related remote information jaise ki social media profiles/content fetch karne ka tarika. Dhyaan dein ki aap profile image ko include karein remote file aur crop/resize support ke saath.

Example 3: Javascript mein B*tree algorithm ko implement karo.

Example 4: GoLang mein Djikstra's algorithm ka solution likho.

Example 5. Koi bhi graphical libraries ke bina, Python mein ek ascii snake game banayo.

Example 6. Python ka use karke ek text-based solitaire implementation banayo. Yahaan kuch additional requirements hain:
 - Game state ko mariadb mein save kiya jaana chahiye.
 - Har move ke baad game ek screenshot lekar record kare.
 - Game sirf keyboard input accept kare, mouse nahi.

Example 7. Niche diye gaye points ke saath ek python script banayo:
 a. Ek single word ko input ke roop mein accept kare.
 b. Agar user ek se zyada word input karta hai, ya word 10 characters se zyada hai, toh ek error message print kare aur stop kar de.
 c. Input word ka bada ascii art version generate kare.

The tasks must be something a coding language model can complete without access to additional resources.

The tasks must be something that could be completed with 2000 words and symbols or less.

Task should be related to language "{programming_language}"

The tasks should be in Hinglish language.

None of the tasks should be about reading from or writing to csvs.

Give me a numbered list of 5 new coding tasks in hinglish.

Tasks should all details required to write the code, including programing language to use or any other data required to completed the task. 

Don't mention the word "hinglish" in tasks generated.

Response format:
TSK 1. [task 1 in hinglish]
TSK 2. [task 2 in hinglish]
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

    base_repo = "manishiitg/indic-synthetic-code"
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
    topics_generated_map = {}
    if repo_exists(base_repo, repo_type="dataset"):
        existing_ds = load_dataset(base_repo, split="train", cache_dir="temp-" + str(time.time()))
        for r in existing_ds:
            final_data.append(r)
            if r["language"] not in topics_generated_map:
                topics_generated_map[r["language"]] = []
            topics_generated_map[r["language"]].append(r["topic"])

    languages = ["hinglish", "hindi"]
    for lang in languages:
        topics_generated = []
        if lang in topics_generated_map:
            topics_generated = topics_generated_map[lang]
        args.lang = lang
        topic_instruct_map = {}

        for loop in range(10):

            prompts = []
            topics_selected = []

            PROGRAMMING_TOPICS = [
                "python",
                "javascript",
                "java",
                "c",
                "c++",
                "golang",
                "C#",
                "bash",
                "powershell",
                "SQL",
            ]
            if args.generate_topics:
                message = []
                prompt = """
                    Give me a numbered list of 50 completely random topics related to programing, coding for which code can be written.
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

                user = USER_PROMPT.replace("{programming_language}", topic_selected)
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

                    system_message_selected = random.choice(SYSTEM_MESSAGES_ORCA)
                    if args.lang == "hindi":
                        system_message_selected += "\n\nAnswer in hindi only. The code blocks should be in english, comments and explanations should be in hindi only."
                    if args.lang == "hinglish":
                        system_message_selected += "\n\nAnswer in hinglish only. Add inline code comments as well.  Translate to hinglish if required."

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
