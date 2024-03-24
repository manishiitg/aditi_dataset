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
You are asked to come up with a set of 25 diverse task instructions and corrosponding input. 

You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.

These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions based on the input.

The instruction should only be related to SUBJECT_AREA

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be related to only SUBJECT_AREA
    a. The type of instruction should not include poem writing
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, knowledge based questions, classification, editing, etc.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should involve realistic data and should not contain simple placeholders. The instructions should provide substantial content to make the instruction challenging but should ideally not exceed 2 to 3 sentences.
6. Make sure every instruction is in hindi language only. 

Examples of instructions/input to generate:

End of examples
INSTRUCTION: दिखावा करो कि आप 1920 के दशक के एक जासूस हैं जो एक अपराध के गवाह से पूछ रहे हैं। संदिग्ध के बारे में जानकारी जानने के लिए 5 सवाल पूछिए।
INPUT: गवाह का कहना है कि वो संदिग्ध को चुराया हुआ सामान लेकर भागते हुए देखा।

INSTRUCTION: कल्पना कीजिए कि आप जेन गुडॉल हैं और एक व्यक्ति से समझते हैं जो अभी एक बड़ा क्षेत्र वनों को ख़त्म कर दिया है कि प्रकृति के निवास स्थलों को सुरक्षित रखने की महत्ता है।
INPUT: वनों को सुरक्षित रखने की क्यों चिंता करनी चाहिए?

INSTRUCTION: आप टेस्ला के एआई असिस्टेंट हैं जो 2040 में बन गया है, जो ग्राहकों की मदद करने में विशेष है। कृपया मुझे मदद करें जिनके घर में टेस्ला कार को चार्ज करने की समस्या आ रही है।
INPUT: मैंने हाल ही में टेस्ला मॉडल एक्स3 खरीदा है और मुझे अपने गैराज में चार्ज करने के लिए कुछ समस्या आ रही है। चार्ज का केबल कार के चार्जिंग पोर्ट से मजबूत कनेक्शन नहीं बन रहा है, और प्रक्रिया हमेशा विफल हो जाती है। क्या आप मेरी मदद कर सकते हैं?

INSTRUCTION: कल्पना कीजिए कि आप एक सुपरहीरो के लिए एआई असिस्टेंट हैं। कैसे उन्हें मदद करोगे जिसमें एक दुश्मन जो छुप गया है, उसका पता लगाया जा सके?
INPUT: सुपरहीरो: नाइटशेड के पता मुझे ढूंढ दो। उसने हमारी आखिरी मुलाकात के बाद शहर में गायब हो गई।

INSTRUCTION: कल्पना कीजिए कि आप एक सुपरहीरो हैं जिसके समय को नियंत्रित करने की शक्ति है। आप अपने शहर को आने वाले आपदा से कैसे बचाएंगे?
INPUT: एक बड़ा भूखाम्प आपके शहर को लगने वाला है, जिसमें बड़े नुक्सान की संभावना है और अनेक लोगों की जिंदगी को खतरा है। समय को हेरफेर करने की शक्ति वाले सुपरहीरो के रूप में, आपके पास आपके शहर और उसके रहने वालों को बचाने के लिए क्या रानियां होंगी?

List of 25 tasks:

The output format should be:
INSTRUCTION: [first instruction in hindi]
INPUT: [first input's answer in hindi]

INSTRUCTION: [second instruction in hindi]
INPUT: [second instruction's answer in hindi]
"""

PROMPT_2 = """
You are asked to come up with a set of 25 diverse task instructions and corrosponding input. 

You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.

These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions based on the input.

The instruction should only be related to SUBJECT_AREA

Here are the requirements:
1. Generate tasks only in hinglish. Translate tasks generated to hinglish if tasks are in hindi.
2. Try not to repeat the verb for each instruction to maximize diversity.
3. The type of instructions should be diverse. 
    a. The list should include diverse types of tasks like open-ended generation, classification, editing, writing, conversation, trivia, etc.
4. The type of instructions should be related to only SUBJECT_AREA
5. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted
7. The instructions should involve realistic data and should not contain simple placeholders. The instructions should provide substantial content to make the instruction challenging but should ideally not exceed 2 to 3 sentences.


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

List of 25 tasks:

The output format should be:
INSTRUCTION: [first instruction in hinglish]
INPUT: [first input's answer in hinglish]

INSTRUCTION: [second instruction in hinglish]
INPUT: [second instruction's answer in hinglish]

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
    if repo_exists(base_repo, repo_type="dataset"):
        existing_ds = load_dataset(base_repo, split="train", cache_dir="temp-" + str(time.time()))
        for r in existing_ds:
            final_data.append(r)

    global TOPICS

    topics_generated = []

    languages = ["hinglish", "hindi"]
    for lang in languages:
        args.lang = lang
        topic_instruct_map = {}

        for loop in range(10):

            prompts = []
            topics_selected = []
            if args.generate_topics or True:
                message = []
                prompt = """
                    Give me a numbered list of 10 completely random topics
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

            outputs = eval_hf_model(args, model, tokenizer, prompts, .2)
            print(outputs)
            

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

            os.exit(1)
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
