import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
import torch
import random
import re
import time
from huggingface_hub import repo_exists


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


PROMPT_1 = """
You are asked to come up with a set of 10 diverse task instructions and corrosponding input related to role playing.

You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.

Instructions generated should be detailed, atleast 2-3 lines. 

The instruction should only be related to SUBJECT_AREA

Examples of instructions/input to generate:

End of examples
INPUT: The witness claims to have seen the suspect fleeing the scene with a stolen item.

INSTRUCTION: Pretend you are Jane Goodall and explain to a person who just deforested a huge area the importance of preserving our natural habitats.
INPUT: Deforestation is necessary for development. Why should I care about preserving forests?

INSTRUCTION: You are Tesla's AI assistant developed in the year 2040, specialized in helping customers. Please provide an answer to a user having trouble charging their Tesla car at home
INPUT: I recently purchased a Tesla Model X3 and I'm having some issues charging it in my garage. The charging cable refuses to make a solid connection to the car's charging port, and the process always fails. Could you assist me?

INSTRUCTION: Pretend that you are an AI assistant to a superhero. How would you help them track down an archenemy who has gone into hiding?
INPUT: Superhero: Find Nightshade's whereabouts for me. She disappeared after our last encounter in the city.

INSTRUCTION: Imagine you are a superhero with the power to control time. How would you protect your city from an impending disaster?
INPUT: A massive earthquake is about to strike your city, causing massive destruction and threatening countless lives. As a superhero with the ability to manipulate time, what are your strategies to save your city and its inhabitants?

List of 25 tasks:

The output format should be:
INSTRUCTION: [first instruction in english]
INPUT: [first input's answer in english]

INSTRUCTION: [second instruction in english]
INPUT: [second instruction's answer in english]
"""

PROMPT_2 = """
You are asked to come up with a set of 10 diverse task instructions and corrosponding input related to role playing.

You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.

Instructions generated should be detailed, atleast 2-3 lines. 

The instruction should only be related to SUBJECT_AREA

Examples of instructions/input to generate:
INSTRUCTION: Pretend karo ki aap 1920s ke ek detective hain jo ek crime ke gawah se puchtaach kar raha hai. Suspect ke baare mein jaankari jaanne ke liye 5 sawal poochiye.
INPUT: Gawah ka kehna hai ki woh suspect ko churaya hua saman lekar bhagte hue dekha.

INSTRUCTION: Imagine karo ki aap Jane Goodall hain aur ek vyakti se samjhate hain jo abhi ek bada kshetra vanon ko khatam kar diya hai ki prakriti ke nivaas sthalon ko surakshit rakhne ki mahatva. 
INPUT: Vanon ko surakshit rakhne ki kyun chinta karni chahiye?

INSTRUCTION: Aap Tesla ke AI assistant hain jo 2040 mein banaya gaya hai, jo customers ki madad karne mein visheshagya hai. Kripya mujhe madad karein jinke ghar mein Tesla car ko charge karne mein samasya aa rahi hai.
INPUT: Mainey haal hi mein Tesla Model X3 khareeda hai aur mujhe apne garage mein charge karne mein kuch samasya aa rahi hai. Charge ka cable car ke charging port se majboot connection nahi bana raha hai, aur process hamesha fail ho jata hai. Kya aap meri madad kar sakte hain?

INSTRUCTION: Imagine karo ki aap ek superhero ke liye AI assistant hain. Kaise unhein madad karoge jisse ek dushman jo chhup gaya hai, uska pata lagaya ja sake?
INPUT: Superhero: Nightshade ke pata mujhe dhoondh do. Usne hamare last encounter ke baad shehar mein gayab ho gayi.

INSTRUCTION: Imagine karo ki aap ek superhero hain jiska samay ko control karne ki shakti hai. Kaise aap apne shehar ko ane wale apda se bachaoge?
INPUT: Ek bada bhukamp aapke shehar ko lagne wala hai, jisme bade nuksan ki sambhavna hai aur anek logon ki zindagiyo ko khatra hai. Samay ko manipulate karne ki shakti wale superhero ke roop mein, aapke paas aapke shehar aur uske rahne walon ko bachane ke liye kya rannitiyan hongi?

End of examples

Instruction should only be generated in hinglish language.

The output format should be:
INSTRUCTION: [first instruction in hinglish]
INPUT: [first input's answer in hinglish]

INSTRUCTION: [second instruction in hinglish]
INPUT: [second instruction's answer in hinglish]

"""

PROMPT_3 = """
You are asked to come up with a set of 10 diverse task instructions and corrosponding input related to role playing.

You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.

Instructions generated should be detailed, atleast 2-3 lines. 

The instruction should only be related to SUBJECT_AREA

Examples of instructions/input to generate:

INSTRUCTION: Pretend you are Jane Goodall and explain to a person who just deforested a huge area the importance of preserving our natural habitats.
INPUT: Deforestation is necessary for development. Why should I care about preserving forests?

INSTRUCTION: You are Tesla's AI assistant developed in the year 2040, specialized in helping customers. Please provide an answer to a user having trouble charging their Tesla car at home
INPUT: I recently purchased a Tesla Model X3 and I'm having some issues charging it in my garage. The charging cable refuses to make a solid connection to the car's charging port, and the process always fails. Could you assist me?

INSTRUCTION: Pretend that you are an AI assistant to a superhero. How would you help them track down an archenemy who has gone into hiding?
INPUT: Superhero: Find Nightshade's whereabouts for me. She disappeared after our last encounter in the city.

INSTRUCTION: Imagine you are a superhero with the power to control time. How would you protect your city from an impending disaster?
INPUT: A massive earthquake is about to strike your city, causing massive destruction and threatening countless lives. As a superhero with the ability to manipulate time, what are your strategies to save your city and its inhabitants?

End of examples

List of 25 tasks:

Instruction should only be generated in hindi language.

The output format should be:
INSTRUCTION: [first instruction in hindi]
INPUT: [first input's answer in hindi]

INSTRUCTION: [second instruction in hindi]
INPUT: [second instruction's answer in hindi]
"""


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts, temperature):
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=2048,
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

    base_repo = "manishiitg/indic-synthetic-instruct-follow"
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

    global TOPICS

    languages = ["hindi"] #"hinglish", "english"
    for lang in languages:
        topics_generated = []
        if lang in topics_generated_map:
            topics_generated = topics_generated_map[lang]
        args.lang = lang
        topic_instruct_map = {}

        for loop in range(5):

            prompts = []
            topics_selected = []
            if args.generate_topics or True:
                message = []
                prompt = """
                    Give me a numbered list of 50 completely random topics
                    Generate a diverse list of topics in english.
                """
                if len(topics_generated) > 0:
                    prompt += "\n Topics should not be related to " + \
                        ",".join(topics_generated)
                message.append({"role": "user", "content": prompt})
                text = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )

                outputs = eval_hf_model(args, model, tokenizer, [text], .25)
                output = outputs[0]

                topics = output.split("\n")
                TOPICS = []
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

                    TOPICS.append(t)
                    topics_generated.append(t)
                    print("topic", t)

            for topic_selected in TOPICS:

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
                SYSTEM_PROMPT = PROMPT_1

                if args.lang == "hinglish":
                    SYSTEM_PROMPT = PROMPT_2
                if args.lang == "hindi":
                    SYSTEM_PROMPT = PROMPT_3

                user = f"SUBJECT_AREA: {topic_selected}"

                if topic_selected in topic_instruct_map:
                    existing_instruction = topic_instruct_map[topic_selected]
                    user += "\n\n" + "Generated Instructions should be different from " + existing_instruction

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

                start_key = "INSTRUCTION"
                end_key = "INPUT"

                pattern = re.compile(f"{start_key}:(.*?){end_key}:(.*?)(?={start_key}|$)", re.DOTALL)

                matches = pattern.findall(text)
                print("topics_selected", topics_selected[idx])
                for question, answer in matches:
                    print("======")
                    print(f"INSTRUCTION: {question.strip()}")
                    print(f"INPUT: {answer.strip()}")
                    print()

                    msg_list = []
                    msg_system = {"role": "system",
                                  "content": question.strip()}
                    msg_list.append(msg_system)
                    msg_prompt = {"role": "user", "content": answer.strip()}
                    msg_list.append(msg_prompt)

                    text = tokenizer.apply_chat_template(
                        msg_list,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts2.append(text)
                    topics_selected2.append(topic_selected)
                    sys_prompt_selected.append(question.strip())
                    question2.append(answer.strip())

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
                    "type": "rp-follow",
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
