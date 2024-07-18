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
You are a visual question answering expert. You must choose the option that best describes the action the person is about to take from four options of [OPTION] based on the [OBSERVATION], [CAPTION], [QUESTION], and [REASON].
'''
incontext_prompt='''
[OBSERVATION]:The current observation from the first-person perspective, usually the starting frame of the action about to be taken.
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities. Each line represents a description of the process of a completed action. At the beginning of each caption, the #C indicates the image seen from your point of view.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Four candidates for the question, each option representing a possible action about to be taken.
[DESCRIPTION]: Based on the current observation image, describe what activity C is engaged in from the first-person perspective, what objects C's hands are interacting with, whether the task-related objects mentioned in the options are visible, and what their status is if they are visible.
[OPTION REASONING]: Based on the [OBSERVATION], [CAPTION], [QUESTION], and [DESCRIPTION], reason about the [OPTION] and eliminate actions that have already been completed.
[SUMMARY]: Based on the [OBSERVATION] and [OPTION REASONING], choose the most likely action C is about to take.
1. Please pay more attention to hand-object interactions in the current observation, and consider the action C is about to take based on the movement trends.
2. If the option has already been completed, do not choose to repeat the action.
3. If the option you want to choose has a prerequisite that is not yet completed, prioritize choosing the prerequisite option.

I will give you some examples as follow:
{example}
Now, you should first describe the current observation. Secondly, make a [OPTION REASONING] based on [OBSERVATION], [CAPTION], [QUESTION] and [DESCRIPTION], then [SUMMARY], give right number of [OPTION] as [ANSWER], which must range from 0 to 3 and can't be none. Additionally, you need to give me [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You SHOULD answer the question, even given a low confidence.
[OBSERVATION]
The image of current observation is shown.
[CAPTION]
{caption}
[QUESTION]
{question}
[OPTION]
{option}
"OUPUT":
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
        model="claude-3-5-sonnet-20240620",
    )
    return result.choices[0].message.content


def llm_inference(queries, output_dir=None, part=None):
    for file in part:
        uid = file[:-5]
        d = queries[int(uid)]
        try:
            with open('./memory/'+uid+'.json','r') as f:
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
            with open('./example_reasoning.txt','r') as ex:
                example=ex.read()

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
    output_dir = './result_claude/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in tqdm(range(0, 1584)):
        f_name = f"{i}.json"
        if os.path.exists(output_dir + f_name): continue
        llm_inference(
            queries, 
            output_dir = output_dir,
            part = [f_name]
        )
        
if __name__ == "__main__":
    main()