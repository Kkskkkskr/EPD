import collections
import os
import re
import time
import ast
import openai
import pandas as pd
from multiprocessing.pool import Pool
import base64
import json, random
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm
import argparse

openai.api_base = "Your api_base"
openai.api_key = "Your api_key"
# os.environ["http_proxy"] = "http://127.0.0.1:7893"
# os.environ["https_proxy"] = "http://127.0.0.1:7893"


systerm='''
You are a visual question answering expert. Given two answers about the same question and their reasoning steps, you must choose the option that is more reasonable.
'''

incontext_prompt='''
[OBSERVATION]: The current observation from the first-person perspective, usually the starting frame of the action about to be taken.
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities. Each line represents a description of the process of a completed action. At the beginning of each caption, the #C indicates the image seen from your point of view.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Four candidates for the question, each option representing a possible action about to be taken.
"OUTPUT":
[DESCRIPTION]: Textual descriptions of the current observation image.
[OPTION REASONING]: Based on the [OBSERVATION], [CAPTION], and [DESCRIPTION], reason about the [OPTION]. 
[SUMMARY]: Based on the [OBSERVATION] and [OPTION REASONING], choose the most likely action C is about to take.

[OBSERVATION]
The image of current observation is shown.
[CAPTION]
{caption}
[QUESTION]
{question}
[OPTION]
{option}

I will give you the two answers as follow:
{example}
Please first give your reason in [REASON], then output the number of the option corresponding to the answer you choose in [ANSWER], finally give the [CONFIDENCE], same as the format of two answers.

Example Format:
/*If you think "ANSWER_A" is more reasonable, and "ANSWER_A" choose option 3. Then you should output [ANSWER]: 3, the same as "ANSWER_A". */
[REASON]: ......,
[ANSWER]: 3,
[CONFIDENCE]: 5.
Now, please choose the more reasonable option.

"OUTPUT":
'''

def gpt4(prompt, image_path):

    image_file = open(image_path, "rb") 
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    result = openai.ChatCompletion.create(
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }],
        model="gpt-4o",
    )
    return result.choices[0].message.content


def llm_inference(queries, output_dir=None, part=None, example=None):

    for file in part:
        uid = file[:-5]
        d = queries[int(uid)]
        try:
            with open('./caption/'+uid+'.json','r') as f:
                captions=json.load(f)
            caps=''
            caps += captions["Caption"]
            caps = caps.strip("\n")

            opt=''
            que="The task goal is " + d['task_goal'] + ". Considering the progress described in the video caption and the current observation, what action is C about to take?"
            opt+='OPTION 0: '+d['choice_a']+".\n"
            opt+='OPTION 1: '+d['choice_b']+".\n"
            opt+='OPTION 2: '+d['choice_c']+".\n"
            opt+='OPTION 3: '+d['choice_d']+".\n"

            instruction=systerm + "\n" + incontext_prompt.format(example=example, caption=caps, question=que, option=opt)+"\n"
            img_path = "./EgoTest_GPT4_img/" + f"{uid}_" + str(d["current_observation_frame"]) + ".jpg"
            response = gpt4(instruction, img_path)
            
            response_dict = {
                "ANSWER": response,
                "INPUT": instruction
            }
            with open(f"{output_dir}/{uid}.json", "w") as f:
                json.dump(response_dict, f, indent=2)

        except Exception:
            print(f"Error occurs when processing this query: {uid}", flush=True)
            traceback.print_exc()
            break
        else:
            break

def main(): 
    qr_path = "./EgoPlan_test.json"
    js_file = open(qr_path, "r")
    queries = json.load(js_file)
    output_dir = './ensemble_GPT_Claude/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fn1 = "./ans_GPT/GPT.json" 
    fn2 = "./ans_claude/claude.json" 
    fn1 = open(fn1, 'r')
    fn2 = open(fn2, 'r')
    ff1=json.load(fn1)
    ff2=json.load(fn2)
    ans_dict = []
    assert len(ff1) == len(ff2), "not equal len\n"
    for i in tqdm(range(len(ff1))):
        cur_ans = None
        if ff1[i]["label"] == ff2[i]["label"]: 
            cur_ans = ff1[i]["label"]
        else: 
            e1 = "./result_GPT4o/"
            e2 = "./result_claude/"
            f_name = f"{i}.json"
            with open(e1 + f_name,'r') as ef1:
                ex1=json.load(ef1)
            with open(e2 + f_name,'r') as ef2:
                ex2=json.load(ef2)
            ex = "\"ANSWER_A\":\n"+ ex1["ANSWER"] + "\n\n" + "\"ANSWER_B\":\n" + ex2["ANSWER"]
            llm_inference(
                queries, 
                output_dir = output_dir,
                part = [f_name],
                example = ex,
            )

        
if __name__ == "__main__":
    main()