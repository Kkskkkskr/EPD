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
You are a visual question answering expert. Based on the [VIDEO], [OVERALL GOAL] and [COMPLETED STEPS], you can accurately determine the [SUB GOAL] of this video. 
'''
incontext_prompt='''
[VIDEO]: Four frames extracted from a two-second video clip, demonstrating the process of achieving a specific sub-goal. Due to the short duration, typically only a minor action is performed.
[OVERALL GOAL]: The overall goal that is achieved through gradual completion of many sub-goals. Reference information can be provided if the video does not clearly identify the sub-goal.
[COMPLETED STEPS]: Actions already completed before the start of the video, establishing prerequisites for the current sub-goal.
[SUB GOAL]: The specific sub-goal the person in the first-person perspective aims to achieve in the video.

Your task is to output the [SUB GOAL] as a brief phrase based on the [VIDEO], [OVERALL GOAL], and [COMPLETED STEPS].
I will give you an example as follow:

Example 1:
[VIDEO] 
<video>
[OVERALL GOAL]
Add garlic to the food and stir
[COMPLETED STEPS]
none
[SUB GOAL]
"pick up the knife"
/*At the beginning of the video, the knife is initially on the table. Through interaction with the person's hand, the knife's status changes to being held in the hand. As for the other objects, the hand reaches towards the broccoli because it is reaching for the knife, while the garlic remains on the cutting board from start to finish. Therefore, the sub-goal and primary action of this video clip is to pick up the knife.*/

Example 2:
[VIDEO] 
<video>
[OVERALL GOAL]
Store the food in the refrigerator
[COMPLETED STEPS]
cover the container; 
pick up container. 
[SUB GOAL]
"carry containers to the refrigerator"
/*In the video, the first-person perspective is moving towards the refrigerator. Although the last frame shows the person about to open the refrigerator, the sub-goal refers to actions completed within the video, not actions that are about to be completed. Therefore, the sub-goal is to carry containers to the refrigerator.*/

Note that comments within /**/ are only for explanatory purposes and should not be included in the output.
Now, you should determine the sub-goal of this video based on the [VIDEO], [OVERALL GOAL], and [COMPLETED STEPS]. You SHOULD follow the format of example.
1. Observe the changes in the state of objects throughout the video frames. Identify the action that results in an actual change in the state of an object. Do not infer actions that are about to be performed; focus only on actions that are completed within the video frames.
2. Please output the subgoal as a brief phrase and do not output anything else!


[VIDEO]
The four frames are shown.
[OVERALL GOAL]
{task_goal}
[COMPLETED STEPS]
{c_steps}
[SUB GOAL]
'''

def gpt4(prompt, image_paths):
    b64_list = []
    for img_path in image_paths:
        image_file = open(img_path, "rb") 
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        b64_list.append(base64_image)
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
                        "url": f"data:image/jpeg;base64,{b64_list[0]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_list[1]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_list[2]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_list[3]}"
                    }
                }
            ]
        }],
        model="gpt-4o",
    )
    return result.choices[0].message.content


def llm_inference(queries, output_dir=None, part=None):

    for file in part:
        uid = file[:-5]
        d = queries[int(uid)]
        try:
            c_steps = ''
            cap_list = []
            for i in range(len(d["task_progress_metadata"])):
                img_paths = []
                for j in range(4):
                    img_path = "./EgoTest_GPT4_img/" + f"{uid}_{i}_{j}" + ".jpg" 
                    img_paths.append(img_path)
                
                if i != 0:
                    instruction=str(uid)+"\n"+systerm + "\n" + incontext_prompt.format(task_goal = d['task_goal'], c_steps = c_steps)+"\n"
                else:
                    instruction=str(uid)+"\n"+systerm + "\n" + incontext_prompt.format(task_goal = d['task_goal'], c_steps = "none")+"\n"
                response = gpt4(instruction, img_paths)
                cap_list.append(response)
                if i!= len(d["task_progress_metadata"]) - 1: c_steps += response + "; "
                else: c_steps += response + ". "
                print(response)
            
            response_dict = {
                "ANSWER": cap_list,
            }
            with open(f"{output_dir}/{uid}.json", "w") as f:
                json.dump(response_dict, f)

        except Exception:
            print(f"Error occurs when processing this query: {uid}", flush=True)
            traceback.print_exc()
            break
        else:
            break

def cap_2_json(output_dir):
    root_dir = output_dir
    js_dir = f"./memory/"
    for i in range(0, 1584):
        js_name = f"{i}.json"
        js_path = os.path.join(root_dir, js_name)
        js_file = open(js_path, "r")
        c_dict = json.load(js_file)
        str_list = c_dict["ANSWER"]
        caps = ''
        for s in str_list:
            s = s.replace('\n', '')
            if s.find("[") != -1 and s.find("]") != -1:
                id1 = s.find("[")
                id2 = s.rfind("]")
                s = s[:id1] + s[id2 + 1:]
            s = s.replace('\"', '')
            assert s.find("[") == -1 and s.find("]") == -1, f"{i}  []"
            assert s.rfind("\"") == -1, f"{i}  answer"
            if s == str_list[-1]: caps += "#C C " + s + ".\n"
            else: caps += "#C C " + s + ".\n"
        if caps == '': caps = "none" 
        ans_dict = {
            "Video Name": i,
            "Caption": caps,
        }
        if not os.path.exists(js_dir):
            os.makedirs(js_dir)
        ans_path = os.path.join(js_dir, f"{i}.json")
        ans_file = open(ans_path, "w")
        json.dump(ans_dict, ans_file, indent=2)
            
def main():
    qr_path = "./EgoPlan_test.json" 
    js_file = open(qr_path, "r")
    queries = json.load(js_file)
    output_dir = './caption_GPT4o_json/'
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
    cap_2_json(output_dir)
    
if __name__ == "__main__":
    main()