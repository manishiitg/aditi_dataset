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
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    # "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
]


PROMPT_1 = """
I would like your help in producing a chain-of-thought style questions/instructions.

Below are a few examples:

उदाहरण 1: जैकी के पास 3 भाई हैं। हर भाई के पास 2 बहनें हैं। जैकी को कितने बहन हैं? कई संभव उत्तर इस प्रश्न के लिए दें, सुनिश्चित करके प्रत्येक प्रश्न का कदम-दर-कदम विचार आपूर्ति करें। आपने विभिन्न संभावना हल्ले दिए हैं, स्थानांतरित करके सर्वश्रेष्ठ उत्तर को चयन करें उपलब्ध जानकारी के आधार पर।

उदाहरण 2: सूर्य की ओर 5 शिर्ट छाया हो जाती हैं। 4 घंटे लगते हैं। कितने घंटे लग जाएंगे 20 शिर्ट को सूर्य की ओर छाया हो जाएं? लिंक-ऑफ-सों विचार प्रदान करने के लिए कई संभव प्रतिक्रियाएं उत्पन्न करें, तत्परता के आधार पर। सभी उपलब्ध जानकारी, संज्ञान, और सही संतुलन के आधार पर सर्वश्रेष्ठ उत्तर को चयन करें।

Provide a set of 5 new, complex, unique, diverse tasks.

Be sure to include a wide variety of tasks, including tasks that explore ideas of set theory, information theory, parallelism, logic, extrapolation from scientific studies, etc., but also be sure to only include questions that have multiple potentially correct answers.

The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.

All output text should be in hindi, but the exact term "TSK" is a special token that must not be translated.

The output format should be:
TSK 1. [question 1 in hindi language, including instructions to provide multiple solutions, rank, and select best]
TSK 2. [question 2 in hindi language, including instructions to provide multiple solutions, rank, and select best]
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
"""

PROMPT_2 = """
I would like your help in producing a chain-of-thought style questions/instructions.

Below are a few examples:

Example 1: Jakie ke 3 bhai hain. Har bhai ke 2 behen hain. Toh Jakie ki kitni behen hain?  Sawal ke liye kayi possible answers do, aur har ek ke liye step-by-step reasoning bhi provide karo. Possible solutions dene ke baad, accuracy aur completeness ke order mein rank karo, phir available information ke aadhar par best possible output select karo.

Example 2: 5 shirt ko sukhane mein 4 ghante lagta hai. Toh 20 shirt ko kitne ghante lagega sukhane? Use chain-of-thought reasoning to generate several possible responses, phir sabse acchi response ko select karo based on sab available information, intuition, aur likelihood of correctness.

Provide a set of 5 new, complex, unique, diverse tasks.

Be sure to include a wide variety of tasks, including tasks that explore ideas of set theory, information theory, parallelism, logic, extrapolation from scientific studies, etc., but also be sure to only include questions that have multiple potentially correct answers.

The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.

All output text should be in hinglish language, but the exact term "TSK" is a special token that must not be translated.

The output format should be:
TSK 1. [question 1 in hinglish language, including instructions to provide multiple solutions, rank, and select best]
TSK 2. [question 2 in hinglish language, including instructions to provide multiple solutions, rank, and select best]
...

Be sure to include "TSK", untranslated, as a prefix as described in response format.
"""


PROMPT_1_RESPONSE = """
Please provide a chain-of-thought style answer to the provide input.

Below are a few examples of question/response pairs.

Example 1:
Input: Jackie has 3 brothers. Each brother has 2 sisters. How many sisters does Jackie have? Give several possible answers to the question, ensuring you provide step-by-step reasoning for each. After you have provided possible solutions, rank the solutions in order of accuracy and completeness, then select the best possible output based on available information.

Example 1 response:
Solution 1:
The information provided states that "Each brother has 2 sisters." This includes Jackie in the count because she is a sister to her brothers. Considering there are three brothers and each of them has the same two sisters, we can deduce that the siblings consist of 3 brothers and 2 sisters in total. If we remove Jackie from the sibling count (since we are trying to find out how many sisters Jackie herself has), we are left with one other girl in the sibling group.

Final answer: Jackie has 1 sister.

Solution 2:
In this scenario, we can assume a complex family structure, such as half-siblings or step-siblings. "Each brother has 2 sisters" could imply that there are sisters not related to Jackie, like half-sisters or step-sisters. Given this complexity, there could be a situation where Jackie doesn't have any sisters at all.

Final answer: Jackie has 0 sisters.

Solution 3:
The situation could involve a non-traditional family structure where each brother has two sisters, but they are not the same for every brother. It's possible that Jackie is not considered a sister by one or more of her brothers due to reasons like estrangement or legal technicalities. In this case, the number of sisters that Jackie has could be different from the number her brothers have.

Final answer: This would depend on the specifics of the family dynamics, and thus can't be definitively answered.

Ranking of solutions from best to worst:
1. Solution 1: This answer is based on a straightforward interpretation of the provided information and the most common family structures. It uses simple deductive reasoning and doesn't require making additional assumptions.
2. Solution 2: While this scenario is possible, it makes assumptions about more complex family structures, which might not apply in this case. The provided information doesn't suggest any complexity in the family structure.
3. Solution 3: This answer is the most speculative as it relies on assumptions about estrangement or legal technicalities. The likelihood of these circumstances is not suggested by the provided information, so this solution seems the least probable.

Best and final answer: Jackie has 1 sister.


Yahan hai Example 2 ka Hinglish translation:

Input: Dhoop mein 5 kurte sukhane mein 4 ghante lagte hain. 20 kurte sukhane mein kitna samay lagega? Chain-of-thought reasoning ka istemaal karke kuchh sambhavit responses generate kijiye, phir available information, intuition aur correctness ki likelihood ke aadhar par sabse achchha response chuniye.

Response to Example 2:

Solution 1: 
Din ke samay aur baaki kitni dhoop hai isko consider kijiye. Maslan, agar aapke paas din mein sirf 4 ghante ki dhoop baaki hai, toh aap ek hi din mein 20 kurte nahi sukha paaoge agar aap ek baar mein sirf 5 kurte hi sukha sakte hain. Maante hue ki agle din kam se kam 12 ghante dhoop rahegi aur haalat aise hain ki 5 kurte 4 ghanton mein consistently sukh jaayenge, hum ise iss tarah likh sakte hain:
total time = dry time per batch size * number of batches + time delayed due to lack of sunlight  

Is case mein, 5 kurton ke ek batch ko sukhane mein 4 ghante lagte hain, aur batches ki sankhya hai (20 / 5 = 4).

Choonki hum yeh maan rahe hain ki humare paas 12 ghante sukhane ka samay hai, iska matlab hai ki humein (24 ghante mein se 12 ghante ghata kar = 12 ghante) ka delay hua.

Toh total time yeh hoga:
4 * 4 + 12 = 28 ghante.  

Final jawaab: 20 kurte sukhane mein 28 ghante lagenge, agar hum maan lein ki 12 ghante mausam aur dhoop ke haalat ache rahenge aur batch size 5 kurte hogi.

Solution 2:
Diya gaya hai ki 5 kurte sukhane mein 4 ghante lagte hain.

Iska matlab hai ki 1 kurta bhi 4 ghante mein hi sukhega, kyunki yeh task parallelizable hai.

Choonki har kurta individually parallel mein sukh raha hai, toh drying time nahi judi. Iska matlab hai ki chahe hum ek saath kitne bhi kurte sukha rahe hain, jab tak sabhi kurton ko equally environment expose karne ke liye kaafi jagah hai, woh sab 4 ghanton mein hi sukh jaayenge.  

Toh, agar hum yeh maan lein ki woh sab parallel mein sukh rahe hain, aur unhe waise hi haalat mein rakha gaya hai jaise ki pehle ke 5 kurton ko rakha gaya tha, toh 20 kurte sukhane mein bhi 4 hi ghante lagenge.

Final jawaab: Choonki yeh task parallelizable hai, isliye 20 kurte sukhane mein bhi 4 hi ghante lagenge.

Solutions ko best se worst ke order mein rank kiya gaya hai:
1. Solution 2: Yeh jawaab sabse zyada sahi lagta hai kyunki isne di gayi information par straightforward reasoning ki hai, jismein yeh nahi bataya gaya ki space ya dhoop ki availability koi limiting factor hai.
2. Solution 1: Yeh jawaab kam probable hai, kyunki task parallelizable hai, aur humne kai assumptions banayi hain jaise ki kitni dhoop baaki hai, aur kitne ghante aise hain jismein kurte exactly 4 ghanton mein sukh jaate hain.  

Best aur final jawaab: Choonki yeh task parallelizable hai, isliye 20 kurte sukhane mein bhi 4 hi ghante lagenge.


End of examples.

The possible solutions should always have the reasoning first, then the final answer. Don't ever put the final answer first, then reasoning.

Make sure you fully understand each solution before providing a ranking. The position of the solution does not always correspond to it's ranking, i.e. sometimes solution 2 or 3 can be better that solution 1, and therefore the ranking should reflect that. Always rank the solutions based on accuracy, completeness, and probability of being correct, not based on their position in the list of possible solutions.

Be sure to include at least 2, preferably 3 possible solutions.

All output text should be in hinglish language only.
"""

PROMPT_2_RESPONSE = """
Please provide a chain-of-thought style answer to the provide input.

Below are a few examples of question/response pairs.

Example1:
Input: Jackie ke paas 3 bhai hain. Har bhai ke paas 2 bahinein hain. Jackie ki kitni bahinein hain? Is sawaal ke kuchh sambhavit jawaab dijiye, aur har jawaab ke liye step-by-step reasoning bhi dijiye. Jab aap sambhavit solutions de chuke, unhe accuracy aur completeness ke order mein rank kar dijiye, phir available information ke aadhar par sabse achchha jawaab chuniye.

Example 1 ka response:

Solution 1:
Di hui information mein likha hai ki "Har bhai ke paas 2 bahinein hain." Yeh Jackie ko bhi count mein lata hai kyunki woh bhi apne bhaiyon ki bahan hai. Maante hue ki 3 bhai hain aur har ek ke paas wahi 2 bahinein hain, hum yeh nikaalne kar sakte hain ki bhai-bahinon mein 3 bhai aur 2 bahinein hain. Agar hum Jackie ko is bhai-bahin group se nikal dete hain (kyunki hum yeh pata karna chahte hain ki Jackie ki kitni bahinein hain), toh ek aur ladki is group mein bachi rahegi.

Final jawaab: Jackie ki 1 bahan hai.

Solution 2: 
Is scenario mein, hum ek complex family structure ka andaza laga sakte hain, jaisa ki half-siblings ya step-siblings. "Har bhai ke paas 2 bahinein hain" ka matlab ho sakta hai ki woh bahinein Jackie se related nahi hain, jaisa ki half-bahinein ya step-bahinein ho sakti hain. Is complexity ko dekhte hue, aisa ho sakta hai ki Jackie ki bilkul bhi bahan na ho.

Final jawaab: Jackie ki 0 bahinein hain.

Solution 3:
Yeh situation ek non-traditional family structure wali ho sakti hai jahan har bhai ke paas 2 bahinein hain, lekin woh har bhai ke liye alag hain. Ho sakta hai ki Jackie ko kuch bhaiyon ne bahan nahi maana ho, kuch karanon se jaisa ki parisparan ya kaanuni technicalities. Is surat mein, Jackie ki jitni bahinein hain woh uske bhaiyon ki bahinon ki sankhya se alag ho sakti hai. 

Final jawaab: Yeh family dynamics ke specifics par nirbhar karega, isliye iska definitive jawaab nahi diya ja sakta.

Solutions ko best se worst ke order mein rank kiya gaya hai:
1. Solution 1: Yeh jawaab di gayi information ka seedha interpretation hai aur sabse aam family structures par aadharit hai. Isne simple deductive reasoning ka istemaal kiya hai aur kisi bhi additional assumption ki zaroorat nahi padi.
2. Solution 2: Yeh scenario sambhav hai, lekin complex family structures ke baare mein assumptions bana raha hai, jo is case mein laagu na ho. Di gayi information mein family structure ki koi complexity nahi di gayi hai. 
3. Solution 3: Yeh jawaab sabse zyada speculative hai kyunki isne estrangement ya legal technicalities ke baare mein assumptions banayi hain. Di gayi information in parisitiyon ki probability nahi batati, isliye yeh solution sabse kam probable lagta hai.

Best aur final jawaab: Jackie ki 1 bahan hai.


Example2:

Input: Dhoop mein 5 kurte sukhane mein 4 ghante lagte hain. 20 kurte sukhane mein kitna samay lagega? Chain-of-thought reasoning ka istemaal karke kuchh sambhavit responses generate kijiye, phir available information, intuition aur correctness ki likelihood ke aadhar par sabse achchha response chuniye.

Response to Example 2:

Solution 1: 
Din ke samay aur baaki kitni dhoop hai isko consider kijiye. Maslan, agar aapke paas din mein sirf 4 ghante ki dhoop baaki hai, toh aap ek hi din mein 20 kurte nahi sukha paaoge agar aap ek baar mein sirf 5 kurte hi sukha sakte hain. Maante hue ki agle din kam se kam 12 ghante dhoop rahegi aur haalat aise hain ki 5 kurte 4 ghanton mein consistently sukh jaayenge, hum ise iss tarah likh sakte hain:
total time = dry time per batch size * number of batches + time delayed due to lack of sunlight  

Is case mein, 5 kurton ke ek batch ko sukhane mein 4 ghante lagte hain, aur batches ki sankhya hai (20 / 5 = 4).

Choonki hum yeh maan rahe hain ki humare paas 12 ghante sukhane ka samay hai, iska matlab hai ki humein (24 ghante mein se 12 ghante ghata kar = 12 ghante) ka delay hua.

Toh total time yeh hoga:
4 * 4 + 12 = 28 ghante.  

Final jawaab: 20 kurte sukhane mein 28 ghante lagenge, agar hum maan lein ki 12 ghante mausam aur dhoop ke haalat ache rahenge aur batch size 5 kurte hogi.

Solution 2:
Diya gaya hai ki 5 kurte sukhane mein 4 ghante lagte hain.

Iska matlab hai ki 1 kurta bhi 4 ghante mein hi sukhega, kyunki yeh task parallelizable hai.

Choonki har kurta individually parallel mein sukh raha hai, toh drying time nahi judi. Iska matlab hai ki chahe hum ek saath kitne bhi kurte sukha rahe hain, jab tak sabhi kurton ko equally environment expose karne ke liye kaafi jagah hai, woh sab 4 ghanton mein hi sukh jaayenge.  

Toh, agar hum yeh maan lein ki woh sab parallel mein sukh rahe hain, aur unhe waise hi haalat mein rakha gaya hai jaise ki pehle ke 5 kurton ko rakha gaya tha, toh 20 kurte sukhane mein bhi 4 hi ghante lagenge.

Final jawaab: Choonki yeh task parallelizable hai, isliye 20 kurte sukhane mein bhi 4 hi ghante lagenge.

Solutions ko best se worst ke order mein rank kiya gaya hai:
1. Solution 2: Yeh jawaab sabse zyada sahi lagta hai kyunki isne di gayi information par straightforward reasoning ki hai, jismein yeh nahi bataya gaya ki space ya dhoop ki availability koi limiting factor hai.
2. Solution 1: Yeh jawaab kam probable hai, kyunki task parallelizable hai, aur humne kai assumptions banayi hain jaise ki kitni dhoop baaki hai, aur kitne ghante aise hain jismein kurte exactly 4 ghanton mein sukh jaate hain.  

Best aur final jawaab: Choonki yeh task parallelizable hai, isliye 20 kurte sukhane mein bhi 4 hi ghante lagenge.


End of examples.

The possible solutions should always have the reasoning first, then the final answer. Don't ever put the final answer first, then reasoning.

Make sure you fully understand each solution before providing a ranking. The position of the solution does not always correspond to it's ranking, i.e. sometimes solution 2 or 3 can be better that solution 1, and therefore the ranking should reflect that. Always rank the solutions based on accuracy, completeness, and probability of being correct, not based on their position in the list of possible solutions.

Be sure to include at least 2, preferably 3 possible solutions.

All output text should be in hinglish language only.
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

    base_repo = "manishiitg/indic-synthetic-cot"
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

    languages = ["hinglish", "hindi"]
    topic_selected = "cot"
    for lang in languages:
        args.lang = lang
        topic_instruct_map = {}

        for loop in range(20):

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

            user = USER_PROMPT
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

            outputs = eval_hf_model(args, model, tokenizer, prompts, .2)

            print(outputs)

            prompts2 = []
            topics_selected2 = []
            sys_prompt_selected = []
            question2 = []
            for idx, text in enumerate(outputs):
                print("======")
                print("prompt", prompts[idx], "text", text)

                instructions = []
                for instruction in re.findall(
                    r"(?:^|\n)TSK \d+\. (.*?)(?:$|(?=\nTSK \d+\. ))", text, re.DOTALL
                ):
                    instructions.append(instruction)
                    if instruction.startswith("["):
                        instructions = instructions[0:]
                    if instruction.endswith("]"):
                        instructions = instructions[:-1]

                    system_message_selected = random.choice(SYSTEM_MESSAGES_ORCA)
                    if args.lang == "hindi":
                        system_message_selected += "\n\nAnswer in hindi only."
                    if args.lang == "hinglish":
                        system_message_selected += "\n\nAnswer in hinglish only. Translate to hinglish if required."

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
                    "type": "cot",
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
