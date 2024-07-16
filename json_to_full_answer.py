import json 
import os 
import re
import time
ans_t = "GPT"
root_dir = f"./result_GPT4o/"
ans_dict = []
js_dir = f"./ans_{ans_t}"

for i in range(0, 1584):
    js_name = f"{i}.json"
    js_path = os.path.join(root_dir, js_name)
    js_file = open(js_path, "r")
    c_dict = json.load(js_file)
    string_representation = c_dict["ANSWER"]
    string_representation = string_representation.replace('\n', '')
    assert string_representation.find("CONFIDENCE") != -1 or string_representation.find("Confidence") != -1 or string_representation.find("confidence"), f"{i} confidence"
    assert string_representation.rfind("ANSWER") != -1 or string_representation.rfind("Answer") != -1, f"{i} answer"
    s_c = max(string_representation.find("CONFIDENCE"), string_representation.find("Confidence"), string_representation.find("confidence") )
    s_a = max(string_representation.rfind("ANSWER"), string_representation.rfind("Answer"))
    id1 = min(s_a, s_c)
    id2 = max(s_a, s_c)
    if s_a < s_c:
        string_representation = string_representation[:id2]
        string_representation = string_representation[id1: ]
    else:
        string_representation = string_representation[id2:]
    ans = -1
    if string_representation.find("0") != -1:
        ans = 0
    elif string_representation.find("1") != -1:
        ans = 1
    elif string_representation.find("2") != -1:
        ans = 2
    elif string_representation.find("3") != -1:
        ans = 3
    if ans not in [0, 1, 2, 3]: print(i)
    ans_dict.append({
        "sample_id": i,
        "label": chr(int(ans)+ord("A")),
    })

if not os.path.exists(js_dir):
    os.makedirs(js_dir)
ans_path = os.path.join(js_dir, f"{ans_t}.json")
ans_file = open(ans_path, "w")
json.dump(ans_dict, ans_file, indent=2)
                    
                