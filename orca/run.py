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
    # "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    # "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    # "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
]


PROMPT_1 = """
I need you to help me generate math/reasoning tests.

I would like you to provide both the question and the solution.

The solutions should always include the work/intermediate steps used to generate a response.

Here are a few examples:

उदाहरण 1: प्रश्न: एक औसत मनुष्य के दिल की धड़कन एक दिन में कितनी होती है? चरण दर चरण तर्क करें।
उत्तर: औसत मानव हृदय गति लगभग 60 से 100 बीट प्रति मिनट होती है। सरलता के लिए, आइए इन दो मानों का औसत लें, जो 80 बीट प्रति मिनट है।

सबसे पहले, आइए गणना करें कि एक घंटे में कितनी धड़कनें होती हैं। हम जानते हैं कि एक घंटे में 60 मिनट होते हैं, इसलिए:

1 घंटे में दिल की धड़कन = हृदय गति * प्रति घंटा मिनट
  = 80 बीट प्रति मिनट * 60 मिनट
  = 4800 बीट्स

फिर, यह पता लगाने के लिए कि एक दिन में कितनी धड़कनें होती हैं, हमें इस प्रति घंटा दर को एक दिन में घंटों की संख्या से गुणा करना होगा, जो कि 24 है:

1 दिन में दिल की धड़कन = प्रति घंटा दिल की धड़कन * प्रति दिन घंटे
  = 4800 बीट्स * 24 घंटे
  = 115,200 बीट्स

तो, एक औसत मानव हृदय एक दिन में लगभग 115,200 बार धड़केगा। यह एक सरलीकरण है, क्योंकि हृदय गति व्यक्तियों के बीच और विभिन्न परिस्थितियों में व्यापक रूप से भिन्न हो सकती है।

उदाहरण 2: प्रश्न: हर 5 मिनट में कितनी बिजली गिरती है? अपना काम दिखाओ।
उत्तर: ऐसा अनुमान है कि पृथ्वी पर हर सेकंड लगभग 100 बार बिजली गिरती है। यह अनुमान किसी भी समय वैश्विक स्तर पर होने वाले तूफानों की संख्या और बिजली उत्पादन की उनकी विशिष्ट दरों पर आधारित है।

यह गणना करने के लिए कि हर 5 मिनट में कितनी बिजली गिरती है, हमें पहले यह पता लगाना होगा कि 5 मिनट में कितने सेकंड होते हैं।

हम वह जानते हैं:
1 मिनट = 60 सेकंड

इसलिए,
5 मिनट = 5 * 60 = 300 सेकंड

यदि पृथ्वी पर प्रति सेकंड लगभग 100 बिजली गिरने का अनुभव होता है, तो 5 मिनट (या 300 सेकंड) में हमलों की संख्या होगी:
प्रहारों की संख्या = प्रहार प्रति सेकंड * सेकंडों की संख्या
 = 100*300
 = 30,000

तो, यह अनुमान लगाया गया है कि पृथ्वी पर हर 5 मिनट में लगभग 30,000 बिजली गिरती हैं।

कृपया ध्यान दें कि यह एक अनुमान है, क्योंकि वास्तविक संख्या विभिन्न कारकों जैसे कि मौसम, दिन का समय और भौगोलिक स्थिति के आधार पर भिन्न हो सकती है।

उदाहरण 3: प्रश्न: कमरे में तीन हत्यारे हैं। कोई अंदर आता है और उनमें से एक को मार देता है। कोई भी कमरा नहीं छोड़ता. कमरे में कितने हत्यारे हैं? चरण-दर-चरण तर्क प्रदान करें.
उत्तर: आइए इस पर चरण दर चरण विचार करें:

1. हम कमरे में तीन हत्यारों से शुरुआत करते हैं।
2. एक अन्य व्यक्ति कमरे में प्रवेश करता है, जिससे कुल लोगों की संख्या चार हो जाती है (तीन मूल हत्यारे और एक नया व्यक्ति)।
3. फिर यह नया व्यक्ति मूल हत्यारों में से एक को मार देता है। यह कृत्य उन्हें भी हत्यारा बना देता है, भले ही मूल हत्यारों में से एक अब मर चुका है।
4. तो, अब हमारे कमरे में तीन लोग जीवित बचे हैं: दो मूल हत्यारे और नया व्यक्ति (जो अपने कृत्य से हत्यारा बन गया है)।
5. इसलिए, सभी हत्यारों (चाहे जीवित हों या मृत) पर विचार करते समय, हमारे पास हैं: दो शेष मूल हत्यारे, नया हत्यारा, और मूल हत्यारा जो नए व्यक्ति द्वारा मारा गया था।

जवाब है चार हत्यारे.

उदाहरण 4: प्रश्न: यदि हम 5 शर्ट सूखने के लिए धूप में रखते हैं, और उन्हें सूखने में 4 घंटे लगते हैं, तो 20 शर्ट सूखने में कितना समय लगेगा? इसे ऐसे समझाएं जैसे मैं पांच साल का हूं।
उत्तर: आइए समस्या का कारण समझें:

1. शर्ट को धूप में सूखने में लगने वाला समय आमतौर पर शर्ट की संख्या पर निर्भर नहीं करता है। बल्कि, यह अन्य कारकों के अलावा सूरज की तीव्रता और शर्ट के गीलेपन पर निर्भर करता है।
2. इसलिए, यदि हमने 5 शर्टें बिछाईं और उन्हें सूखने में 4 घंटे लगे, तो यदि हम एक ही समय में 20 शर्टें बिछाते हैं, तो उन्हें सूखने में भी वही 4 घंटे लगेंगे, यह मानते हुए कि सूरज की तीव्रता और अन्य स्थितियाँ बनी रहेंगी वही।
3. ऐसा इसलिए है क्योंकि सभी शर्ट एक के बाद एक नहीं बल्कि एक साथ सूख रही हैं। यदि आप एक के बाद एक शर्ट सुखा रहे थे, तो आपको प्रत्येक शर्ट के सूखने के समय से शर्ट की संख्या को गुणा करना होगा।
4. हालाँकि, इस परिदृश्य में, शर्ट एक ही समय में सूख रही हैं, इसलिए अधिक शर्ट के लिए अतिरिक्त समय की आवश्यकता नहीं है।

इसलिए, मूल 5 शर्ट जैसी ही परिस्थितियों में एक साथ बिछाने पर 20 शर्ट को सुखाने में अभी भी 4 घंटे लगते हैं।

उदाहरण 5: प्रश्न: जेन, जो से तेज़ है। जो सैम से तेज़ है। क्या सैम जेन से तेज़ है? अपनी विचार प्रक्रिया स्पष्ट करें.
उत्तर: इसे समझने का एक आसान तरीका यहां दिया गया है। आइए कल्पना करें कि वे सभी एक दौड़ में हैं।

सबसे पहले, जेन जो के खिलाफ दौड़ती है और जीत जाती है क्योंकि जेन जो से तेज़ है।

इसके बाद, जो सैम के खिलाफ दौड़ लगाता है और जो जीत जाता है क्योंकि जो सैम से तेज है।

इसलिए, यदि जेन जो से तेज़ है, और जो सैम से तेज़ है, तो जेन को सैम से भी तेज़ होना होगा।

इसे ऐसे समझें: जेन, जो से तेज़ दौड़ सकता है, और जो सैम से तेज़ दौड़ सकता है, इसलिए सैम जेन से आगे नहीं निकल पाएगा, जो सबसे तेज़ है।

तो अंतिम उत्तर है नहीं, सैम जेन से तेज़ नहीं है।

उदाहरण 6: प्रश्न: जैकी के 3 भाई हैं। प्रत्येक भाई की 2 बहनें हैं। जैकी की कितनी बहनें हैं? अपना तर्क शामिल करें.
उत्तर: इसके पीछे तर्क यह है:

1. हम जानते हैं कि जैकी के 3 भाई हैं।
2. फिर हमें बताया गया कि उसके प्रत्येक भाई की 2 बहनें हैं।
3. जैकी अपने भाइयों की बहनों में से एक है।
4. तो, यदि उसके प्रत्येक भाई की 2 बहनें हैं, तो इसका मतलब है कि जैकी के अलावा एक और बहन होनी चाहिए।
5. अत: जैकी की 2 बहनें हैं।

यह ध्यान रखना महत्वपूर्ण है कि प्रत्येक भाई की बहनों की संख्या में स्वयं जैकी भी शामिल है, इसलिए केवल एक अतिरिक्त बहन क्यों है। प्रश्न के वाक्यांश से यह आम ग़लतफ़हमी पैदा हो सकती है कि प्रत्येक भाई की जैकी के अलावा 2 बहनें भी हैं, लेकिन यहाँ ऐसा नहीं है।

अंतिम उत्तर: जैकी की 2 बहनें हैं।

End of examples.

Please write {batch_size} more high quality examples of a similar format, with "QUESTION: [question]" and "ANSWER: [answer]" with the correct answer.

Make sure the questions are somewhat difficult and not trivial. Don't add line separators or use any special formatting (e.g. markdown) in the output.

Make sure each answer includes the steps/work used to generate the response.

The provided examples are very easy; the new questions should be much more difficult to solve.

The new questions must not be simple formulas, i.e.:
- do not ask about anything related to volume, area, circumference, perimeter or other measurements of shapes
- do not ask about about anything related to distances traveled or speed calculations
- do not ask about anything related to trains or speed

The answers should always have the reasoning first, then the final answer. Don't ever put the final answer first, then reasoning.

The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.

All output text should be in hindi, but the exact terms "QUESTION" and "ANSWER" are special tokens that must not be translated.

All questions generated should be related to topic "{topic_selected}"  and generate {batch_size} number of questions.

The output format should be:
QUESTION: [first question in hindi]
ANSWER: [first question's answer in hindi]

QUESTION: [second question in hindi]
ANSWER: [second question's answer in hindi]
"""

PROMPT_2 = """
I need you to help me generate math/reasoning tests.

I would like you to provide both the question and the solution.

The solutions should always include the work/intermediate steps used to generate a response.

Here are a few examples:

Example 1: 
QUESTION: Ek average insaan ke dil ki dhadkane ek din me kitni hoti hai? Step-by-step wajah.

ANSWER: Ek average insaan ka dil ka dhadkan ka rate lagbhag 60 se 100 dhadkan prati minute hota hai. Aasaani ke liye, chalo in do values ka average lete hai, jo ki 80 dhadkan prati minute hai.

Pehle, chalo calculate karte hai ki ek ghante me kitni dhadkane hoti hai. Hum jaante hai ki ek ghante me 60 minute hote hai, to:

Dil ki dhadkane 1 ghante me = Dil ki dhadkan dar * Minute prati ghanta
= 80 dhadkan prati minute * 60 minute
= 4800 dhadkan

Phir, ye jaanne ke liye ki ek din me kitni dhadkane hoti hai, hume is ghante ke rate ko din ke ghanton se guna karna hoga, jo ki 24 hai:

Dil ki dhadkane 1 din me = Dhadkane prati ghanta * Ghante prati din
= 4800 dhadkan * 24 ghante
= 115,200 dhadkan

To, ek average insaan ka dil lagbhag 115,200 baar ek din me dhadakta hai. Ye ek saralikaran hai, kyonki alag-alag vyaktiyon mein aur alag paristhitiyon ke neeche dil ki dharakne ki dar kaafi vari kar sakti hai.

Example 2: QUESTION: Har 5 minute me kitne lightning strikes hote hain? Apna kaam dikhaye.
ANSWER: Anuman lagaya jata hai ki Prithvi ko har second lagbhag 100 baar lightning strike karti hai. Yeh anuman globally kisi bhi samay thunderstorms ki sankhya aur unke typical rates of lightning production par aadharit hai.

Har 5 minute me kitne lightning strikes hote hain, iska calculation karne ke liye, hume pehle yeh figure out karna padega ki 5 minute me kitne seconds hote hain.

Hum jante hain ki:
1 minute = 60 seconds

To,
5 minute = 5 * 60 = 300 seconds

Agar Prithvi ko lagbhag har second 100 lightning strikes milte hain, to 5 minute (ya 300 seconds) me strikes ki sankhya hogi:
Number of strikes = Strikes per second * Number of seconds
 = 100 * 300
 = 30,000

To, anumanit taur par Prithvi par har 5 minute me lagbhag 30,000 lightning strikes hote hain.

Kripya dhyan dein ki yeh ek anuman hai, kyonki vastavik sankhya ritu, din ka samay, aur bhugolik sthiti jaise vibhinn factors par nirbhar kar sakti hai.

Example 3: QUESTION: Kamre me teen killers hain. Koi vyakti andar aata hai aur unme se ek ko maar deta hai. Koi vyakti kamre se bahar nahi jata. Kamre me kitne killers hain? Kadam dar kadam tark prastut karein.
ANSWER: Chaliye isko kadam dar kadam sochte hain:

1. Hum teen killers ke saath shuru karte hain jo kamre me hain.
2. Ek aur vyakti kamre me pravesh karta hai, jisse kul vyaktiyon ki sankhya char ho jati hai (teen mool killers aur ek naya vyakti).
3. Yeh naya vyakti phir mool killers me se ek ko maar deta hai. Is karya se vo bhi ek killer ban jata hai, halaki ab ek mool killer mar chuka hai.
4. To, ab humare paas kamre me teen log bache hain jo jeevit hain: mool killers me se do aur naya vyakti (jo unke karya ke dwara ek killer ban gaya hai).
5. Isliye, jab hum sabhi killers ko consider karte hain (chahe vo jeevit ho ya mar chuke ho), humare paas hain: bache huye do mool killers, naya killer, aur vo mool killer jo naye vyakti ke dwara mara gaya tha.

Jawab hai char killers.

Example 4: QUESTION: Agar hum 5 shirts ko sookhne ke liye dhoop me rakh de, aur unhe sookhne me 4 ghante lagte hain, to 20 shirts sookhne me kitna samay lagega? Isse aise samjhayein jaise main paanch saal ka hoon.
ANSWER: Chaliye problem par vichar karte hain:

1. Shirts ko dhoop me sookhne me samay ka aadhar shirts ki sankhya par aam taur par nahi hota. Balki, yeh dhoop ki tezness aur shirts ki gilapan, sahit anya factors par nirbhar karta hai.
2. To, agar humne 5 shirts rakh di aur unhe sookhne me 4 ghante lag gaye, to agar hum ek hi samay me 20 shirts rakh de, to vo bhi sookhne me same 4 ghante lenge, maan lijiye ki dhoop ki tezness aur anya conditions same rehti hain.
3. Yeh isliye hai kyunki sabhi shirts ek saath sookh rahi hain, ek ke baad ek nahi. Agar aap shirts ko ek ke baad ek sookh rahe hote, to aapko har shirt ke sookhne ke samay ko shirts ki sankhya se guna karna padta.
4. Halaki, is scenario me, shirts ek saath sookh rahi hain, to adhik shirts ke liye koi atirikt samay ki avashyakta nahi hoti hai.

Atah, yeh abhi bhi 4 ghante lagte hain 20 shirts sookhne me jab vo ek saath rakh di jati hain same conditions me jaise mool 5 shirts.

Example 5: QUESTION: Jane Joe se tez hai. Joe Sam se tez hai. Kya Sam Jane se tez hai? Apna vichar prakriya samjhaye.
ANSWER: Isko samajhne ka ek aasan tareeka hai. Chaliye kalpana karte hain ki ve sab ek race me hain.

Pehle, Jane Joe ke khilaf race karti hai aur jeet jati hai kyonki Jane Joe se tez hai.

Agla, Joe Sam ke khilaf race karta hai aur Joe jeet jata hai kyonki Joe Sam se tez hai.

To, agar Jane Joe se tez hai, aur Joe Sam se tez hai, to Jane Sam se bhi tez hona chahiye.

Is tarah sochiye: Jane Joe se tez daud sakti hai, aur Joe Sam se tez daud sakta hai, to Sam Jane ko outrun nahi kar payega, jo ki sabse tez hai.

To antim uttar hai nahi, Sam Jane se tez nahi hai.

Example 6: QUESTION: Jackie ke 3 bhai hain. Har bhai ke 2 behen hain. Jackie ki kitni behenein hain? Apni tark bhi shaamil karein.
ANSWER: Iske peeche ki tark yeh hai:

1. Hum jante hain ki Jackie ke 3 bhai hain.
2. Phir hume bataya jata hai ki uske har bhai ke 2 behenein hain.
3. Jackie apne bhaiyon ki ek behen hai.
4. To, agar uske har bhai ke 2 behenein hain, to iska matlab hai ki Jackie ke alawa ek aur behen honi chahiye.
5. Isliye, Jackie ki 2 behenein hain.

Yeh dhyaan dena mahatvapoorn hai ki har bhai ki behenon ki sankhya me Jackie khud shaamil hai, isliye sirf ek additional behen hai. Prashna ke phrasing se ek aam galatfahmi ho sakti hai ki har bhai ke paas Jackie ke alawa 2 behenein hain, lekin yahan aisa nahi hai.

Antim uttar: Jackie ki 2 behenein hain.

End of examples.

Please write {batch_size} more high quality examples of a similar format, with "QUESTION: [question]" and "ANSWER: [answer]" with the correct answer.

Make sure the questions are somewhat difficult and not trivial. Don't add line separators or use any special formatting (e.g. markdown) in the output.

Make sure each answer includes the steps/work used to generate the response.

The provided examples are very easy; the new questions should be much more difficult to solve.

The new questions must not be simple formulas, i.e.:
- do not ask about anything related to volume, area, circumference, perimeter or other measurements of shapes
- do not ask about about anything related to distances traveled or speed calculations
- do not ask about anything related to trains or speed

The answers should always have the reasoning first, then the final answer. Don't ever put the final answer first, then reasoning.

The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  The response must not contain any notes or information about Flesch-Kincaid scores.

All output text should be in hinglish language, but the exact terms "QUESTION" and "ANSWER" are special tokens that must not be translated.

All questions generated should be related to topic "{topic_selected}" and generate {batch_size} number of questions.

The output format should be:
QUESTION: [first question in hinglish]
ANSWER: [first question's answer in hinglish]

QUESTION: [second question in hinglish]
ANSWER: [second question's answer in hinglish]
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

    base_repo = "manishiitg/indic-synthetic-orca"
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

    languages = ["hindi", "hinglish"] #""
    topic_selected = "orca"
    PROGRAMMING_TOPICS = []
    for lang in languages:
        args.lang = lang
        topics_generated = []
        if lang in topics_generated_map:
            topics_generated = topics_generated_map[lang]
        topic_instruct_map = {}

        for loop in range(20):
            prompts = []
            if args.generate_topics or True:
                message = []
                prompt = """
                    Give me a numbered list of 50 completely random topics related to common sense reasoning, maths.
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
                if len(existing_instructions) > 50:
                    topic_instruct_map[topic_selected] = ",".join(existing_instructions[:50])
                else:
                    topic_instruct_map[topic_selected] = ",".join(existing_instructions)

                msg_list = []
                USER_PROMPT = PROMPT_1

                if args.lang == "hinglish":
                    USER_PROMPT = PROMPT_2

                # if topic_selected in topic_instruct_map:
                #     existing_instruction = topic_instruct_map[topic_selected]
                #     if len(existing_instruction) > 0:
                #         USER_PROMPT += "\n\n" + "Generated questions should be different from " + existing_instruction

                user = USER_PROMPT.replace("{batch_size}", "5")
                user = user.replace("{topic_selected}", topic_selected)
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

                outputs = eval_hf_model(args, model, tokenizer, prompts, 0)

                for idx, text in enumerate(outputs):
                    # print("prompt", prompts[idx], "text", text)

                    start_key = "QUESTION"
                    end_key = "ANSWER"

                    pattern = re.compile(f"{start_key}:(.*?){end_key}:(.*?)(?={start_key}|$)", re.DOTALL)

                    matches = pattern.findall(text)

                    for question, answer in matches:
                        print("======")
                        print(f"QUESTION: {question.strip()}")
                        print(f"ANSWER: {answer.strip()}")
                        print()

                        final_data.append({
                            "topic": topic_selected,
                            "question": question,
                            "answer": answer,
                            "system_prompt": "",
                            "language": args.lang,
                            "type": "orca",
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
