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
    # "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant.",
    # "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    # "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
]


PROMPT_1 = """
Here are a few example prompts:
उदाहरण 1: रवीन्द्रनाथ टैगोर की शैली में मोर के बारे में एक कविता लिखें।
उदाहरण 2: कल्पना कीजिए कि आप फिल्म अग्निपथ में अमिताभ बच्चन का किरदार विजय दीनानाथ चौहान हैं। उन्हीं की शैली में भ्रष्टाचार की भर्त्सना करते हुए एक भाषण लिखिए।
उदाहरण 3: अकबर और बीरबल के बीच एक विनोदी संवाद बनाएं, जिसमें बीरबल अपनी बुद्धि का उपयोग करके अकबर के चुनौतीपूर्ण प्रश्नों में से एक का उत्तर दे।
उदाहरण 4: एक युवा को सलाह देने वाले एक बुद्धिमान वृद्ध साधु के शब्दों और शैली का उपयोग करके "धर्म" का अर्थ समझाएं।
उदाहरण 5: कल्पना कीजिए कि आप दिलवाले दुल्हनिया ले जाएंगे में शाहरुख खान का किरदार राहुल हैं। उन्हीं के अंदाज में लंदन जाने से पहले सिमरन को अपनी भावनाएं व्यक्त करते हुए एक पत्र लिखें।
उदाहरण 6: आप रानी लक्ष्मीबाई के रूप में अपने सैनिकों को झाँसी की रक्षा के लिए प्रेरित करते हुए एक वीर रस से परिपूर्ण भूषण लिखिए, जिसमें आप उन्हें अंग्रेज़ शसक के अत्याचार के ख़िलाफ़ लड़ने के लिए तैयार करते हैं।

उदाहरण 7: सूर्यकांत त्रिपाठी 'निराला' के अंदाज में प्रकृति के सौंदर्य और उसके मानव जीवन पर प्रभाव को दर्शाते हुए एक लंबी कविता लिखें, जिसमें आप वसंत ऋतु के आगमन का वर्णन करते हैं।

उदाहरण 8: आप चाणक्य के चरित्र को दर्शाते हुए एक राजनीतिक गुरु के रूप में अपने शिष्य को राज्य चलाने के कला पर सलाह देते हुए एक संवाद लिखिए, जिसमें आप कुटिलता और धोखेबाज़ी से बचना सीख देते हैं।

उदाहरण 9: आप भगत सिंह के किरदार में इंकलाब जिंदाबाद के नारे के साथ एक जज़्बाती भाषा लिखिए, जिसमें आप भारत की स्वतंत्रता के लिए अपने प्राण न्योछावर करने की बात करते हैं और देश के युवाओं को क्रांतिकारी बनने के लिए प्रेरित करते हैं।

उदाहरण 10: तुलसीदास की शैली में राम और सीता के वनवास के समय के एक प्रसंग पर आधार एक पद लिखिए, जिसमें आप पवित्र प्रेम और त्याग की भावना को दर्शाते हैं।

Generate a set of {batch_size} new similar prompts.

Be appropriately loquacious for the task, e.g. stories should be long, complex, and detailed, whereas a haiku should be the standard three/5-7-5 format.

Make sure to generate diverse and detailed tasks. 
Every task should be atleast 2-3 lines. 

All output task should be in hindi language.

Response format:
TSK 1. task 1 in hindi
TSK 2. task 2 in hindi
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
"""

PROMPT_2 = """
Here are a few example prompts:
TSK 1: Rabindranath Tagore ke andaaz mein peacocks ke baare mein ek kavita likhiye.

TSK 2: Aap Amitabh Bachchan ke character Vijay Deenanath Chauhan se imagine karo, film Agneepath se. Uske andaaz mein, bhrashtachar ke khilaaf ek bhashan likhiye.  

TSK 3: Akbar aur Birbal ke beech ek haasya-paripurna samvad likhiye, jismein Birbal apni buddhi se Akbar ke ek kathin sawal ka uttar deta hai.

TSK 4: "Dharma" ka arth ek budhhe sadhu (sant purush) ke shabd aur andaaz ka prayog karke samjhaiye, jaisa ki woh ek yuvak ko salah de raha ho.

TSK 5: Aap Shah Rukh Khan ke character Rahul se imagine karo, film Dilwale Dulhania Le Jayenge se. Uske andaaz mein, London jaane se pehle Simran ko apne jazbaat bayaan karte hue ek patra likhiye.

TSK 6: Aap Rani Lakshmibai ke roop mein apne sainikon ko Jhansi ki raksha ke liye prerit karte hue ek veer ras se paripurna bhashan likhiye, jismein aap unhe angrez shasak ke atyachar ke khilaaf ladne ke liye taiyaar karte hain.

TSK 7: Suryakant Tripathi 'Nirala' ke andaaz mein prakriti ke saundarya aur uske manav jeevan par prabhaav ko darshate hue ek lambi kavita likhiye, jismein aap vasant ritu ke aagman ka varnan karte hain.

TSK 8: Aap Chanakya ke charitra ko darshate hue ek rajneetik guru ke roop mein apne shishya ko rajya chalane ke kala par salah dete hue ek samvaad likhiye, jismein aap kutilta aur dhokhadadi se bachne ki sikh dete hain.

TSK 9: Aap Bhagat Singh ke kirdar mein Inquilab Zindabad ke naare ke saath ek jazbati bhashan likhiye, jismein aap Bharat ki swatantrata ke liye apne pran nyochhawar karne ki baat karte hain aur desh ke yuvaaon ko krantikari banne ke liye prerit karte hain.

TSK 10: Tulsidas ke shaili mein Ram aur Sita ke vanvaas ke samay ke ek prasang par aadhaarit ek pad likhiye, jismein aap pavitra prem aur tyaag ki bhaavna ko darshate hain.

End of examples. 

Generate a set of {batch_size} new similar prompts.

Be appropriately loquacious for the task, e.g. stories should be long, complex, and detailed, whereas a haiku should be the standard three/5-7-5 format.

Make sure to generate diverse writing tasks. 
Every task should be atleast 2-3 lines. 

All output task should be in hinglish language.

Response format:
TSK 1. task 1 in hinglish language
TSK 2. task 2 in hinglish language
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

    base_repo = "manishiitg/indic-synthetic-roleplay"
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

    languages = ["hinglish","hindi"]
    topic_selected = "roleplay"
    for lang in languages:
        args.lang = lang
        topic_instruct_map = {}

        for loop in range(10):

            prompts = []

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
            USER_PROMPT = PROMPT_1

            if args.lang == "hinglish":
                USER_PROMPT = PROMPT_2

            if topic_selected in topic_instruct_map:
                existing_instruction = topic_instruct_map[topic_selected]
                if len(existing_instruction) > 0:
                    USER_PROMPT += "\n\n" + "Generated Tasks should be different from " + existing_instruction

            user = USER_PROMPT.replace("{batch_size}", "10")
            SYSTEM_PROMPT = "You are an helpful AI assistant."
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

            outputs = eval_hf_model(args, model, tokenizer, prompts, .2)

            prompts2 = []
            topics_selected2 = []
            sys_prompt_selected = []
            question2 = []
            for idx, text in enumerate(outputs):
                # print("prompt", prompts[idx], "text", text)

                instructions = []
                for instruction in re.findall(
                    r"(?:^|\n)TSK \d+\. (.*?)(?:$|(?=\nTSK \d+\. ))", text, re.DOTALL
                ):
                    instructions.append(instruction)
                    if instruction.startswith("["):
                        instructions = instructions[0:]
                    if instruction.endswith("]"):
                        instructions = instructions[:-1]
                    

                    print("=======")
                    print(instruction)

                    system_message_selected = random.choice(SYSTEM_MESSAGES_ORCA)
                    if args.lang == "hindi":
                        system_message_selected += "\n\nAnswer in hindi only."
                    if args.lang == "hinglish":
                        system_message_selected += "\n\nAnswer in hinglish only."

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

            outputs2 = eval_hf_model(args, model, tokenizer, prompts2, 0)
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
                    "type": "roleplay",
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
