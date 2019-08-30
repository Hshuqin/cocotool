import numpy as np
import os
import json
from PIL import Image

def load_func(fpath):#：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

coco_path = 'crowdhuman2coco_val.json'
cocoval = load_func(coco_path)
cocoimages = cocoval[0]["images"]
cocoann = cocoval[0]["annotations"]
cococat = cocoval[0]["categories"]

results = []

for i in range(len(cocoimages)):#len(cocoimages)
    result = {"fpath":cocoimages[i]["file_name"],"gtboxes":[],"width":cocoimages[i]["width"],"height":cocoimages[i]["height"],"ID":cocoimages[i]["id"]}
    results.append(result)

for i in range(len(results)):    #len(results)
    for j in range(len(cocoann)):
        if results[i]["ID"] == cocoann[j]["image_id"]:
            if cocoann[j]["category_id"] == 1:  # people
                gtbox = {"tag":"people","box":cocoann[j]["bbox"],"extra":{"ignore":cocoann[j]["ignore"]}}
                results[i]["gtboxes"].append(gtbox)
            else:
                gtbox = {"tag":"mask","box":cocoann[j]["bbox"],"extra":{"ignore":cocoann[j]["ignore"]}}
                results[i]["gtboxes"].append(gtbox)

f = open('coco2human.odgt','a')
for i in range(len(results)):    #len(results)
    s = json.dumps(results[i])
    f.writelines(s)
    if i != len(results) - 1:
        f.writelines('\n')
f.close()
print(len(results))