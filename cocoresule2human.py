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

coco_path = 'result_64.pkl.bbox.json'
records = load_func(coco_path)   #提取odgt文件数据
records = records[0]             #取出所有数据，是一个list，长度104841
#转换成字典的格式，所有字典里存放在list中。
results = []                     #最终转化的结果，还需将数据放入文件
k = 0
print(len(records))               #  104841
print(records[0]["image_id"])     #  1

for i in range(1,4371):           #  len(images)  
    ann = {"height":0,"ID":0,"width":0,"dtboxes":[]}
    for j in range(len(records)):
        if records[j]["image_id"] == i:
            dtbox = {"score":0,"tag":0,"box":[]}
            ann["ID"] = records[j]["image_id"]
            dtbox["box"] = records[j]["bbox"]
            dtbox["tag"] = records[j]["category_id"]
            dtbox["score"] = records[j]["score"]
            ann["dtboxes"].append(dtbox)
            
    results.append(ann)


# 从val文件中，先匹配id，然后找出他们的宽和高。
cocoeval_path  = 'crowdhuman2coco_val.json'
f_eval = load_func(cocoeval_path)
f_eval = f_eval[0]                # 三个字典，分别是images，annotation，categories
images = f_eval["images"]
print(len(images))

for i in range(len(results)):                
    for j in range(len(images)):
        if(images[j]["id"] == results[i]["ID"]):
            results[i]["height"] = images[j]["height"]
            results[i]["width"] = images[j]["width"]
            continue

print(len(results))
# list to  file
f = open('coco_epoch-64.human','a')
for i in range(len(results)):
    s = json.dumps(results[i])
    f.writelines(s)
    if i != len(results) - 1:
        f.writelines("\n")
f.close()
