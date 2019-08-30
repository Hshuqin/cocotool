import cv2
import os
import json

def load_func(fpath):#：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

human_path = load_func('coco_epoch-30.human')
odgt_path = load_func('coco2human.odgt')
imagesPath = '/home/huangshuqin/mmdetection/data/crowdhuman/Images/'

for i in range(17):
    img_path = imagesPath + odgt_path[i]["fpath"]
    img = cv2.imread(img_path)
    print('gt框数：'+str(len(odgt_path[i]["gtboxes"])))
    print('ID号：'+str(odgt_path[i]["ID"]))
    for j in range(len(odgt_path[i]["gtboxes"])):
        x = odgt_path[i]["gtboxes"][j]["box"][0]
        y = odgt_path[i]["gtboxes"][j]["box"][1]
        w = odgt_path[i]["gtboxes"][j]["box"][2]
        h = odgt_path[i]["gtboxes"][j]["box"][3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    human_pos = -1 # 标记上一张图片，在human_path的位置
    # 在human_path中找到另一张图片的位置
    for k in range(len(human_path)):
        if odgt_path[i]["ID"] == human_path[k]["ID"]:
            human_pos = k
            continue
    # 还是要判断一下，human_pos 不等于 -1
    print('dt框数：'+str(len(human_path[human_pos]["dtboxes"])))
    print('ID号：'+str(human_path[human_pos]["ID"]))
    for k in range(len(human_path[human_pos]["dtboxes"])):
        x1 = human_path[human_pos]["dtboxes"][k]["box"][0]
        y1 = human_path[human_pos]["dtboxes"][k]["box"][1]
        w1 = human_path[human_pos]["dtboxes"][k]["box"][2]
        h1 = human_path[human_pos]["dtboxes"][k]["box"][3]
        cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w1),int(y1+h1)),(0,255,0),2)
    print("-----------------------------------")
    cv2.imwrite('image/'+odgt_path[i]["fpath"],img)

